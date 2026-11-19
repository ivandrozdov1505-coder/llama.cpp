use crate::config::{Backend, EngineConfig, is_model};
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use std::{
    ffi::OsString,
    fs,
    io::{BufRead, BufReader, Read},
    net::{Ipv4Addr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use uuid::Uuid;

pub const MODEL_ID: &str = "dory-local";

#[derive(Clone, Debug)]
pub struct Endpoint {
    pub port: u16,
    pub key: String,
    pub backend: String,
    pub exposed: bool,
    pub lan: bool,
    pub context: u32,
}

impl Endpoint {
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
    }

    pub fn opencode_config(&self, legacy: bool) -> String {
        if !legacy {
            return serde_json::to_string_pretty(&json!({
                "$schema": "https://opencode.ai/config.json",
                "providers": {"dory-local": {
                    "package": "@opencode/ai/providers/openai-compatible",
                    "name": "Dory Workbench",
                    "settings": { "baseURL": self.url(), "apiKey": self.key },
                    "models": { (MODEL_ID): { "name": "Dory Local" } }
                }},
                "model": format!("dory-local/{MODEL_ID}")
            }))
            .expect("JSON serialization");
        }
        serde_json::to_string_pretty(&json!({
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "dory-local": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Dory Workbench",
                    "options": { "baseURL": self.url(), "apiKey": self.key },
                    "models": { (MODEL_ID): {
                        "name": "Dory Local",
                        "limit": { "context": self.context, "output": self.context.min(4096) }
                    }}
                }
            },
            "model": format!("dory-local/{MODEL_ID}")
        }))
        .expect("JSON serialization")
    }
}

#[derive(Debug)]
pub enum EngineEvent {
    Stage(String),
    Log(String),
    Ready(Endpoint),
    Cycle(u32, u32),
    Exited(Result<(), String>),
}

pub struct Engine {
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl Engine {
    pub fn start(
        config: EngineConfig,
        tell: impl Fn(EngineEvent) + Clone + Send + 'static,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let worker = thread::spawn(move || {
            let result = run(config, &flag, &tell).map_err(|e| format!("{e:#}"));
            let result = if flag.load(Ordering::Relaxed) {
                Ok(())
            } else {
                result
            };
            tell(EngineEvent::Exited(result));
        });
        Self {
            stop,
            worker: Some(worker),
        }
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.stop();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn run(
    config: EngineConfig,
    stop: &AtomicBool,
    tell: &(impl Fn(EngineEvent) + Clone + Send + 'static),
) -> Result<()> {
    ensure!(is_model(&config.model), "Выбери локальный GGUF-файл.");
    ensure!(
        config.ubatch <= config.batch,
        "Размер микробатча должен быть не больше батча."
    );
    ensure!(
        config.gpu_layers == "auto"
            || config.gpu_layers == "all"
            || config.gpu_layers.parse::<u32>().is_ok(),
        "Слои GPU: auto, all или число от 0."
    );
    ensure!(
        config.value_cache == "f16" || config.flash != "off",
        "Для квантованного V-кэша включи Flash Attention."
    );
    let (binary, backend) = engine_binary(&config, stop, tell)?;
    if stop.load(Ordering::Relaxed) {
        return Ok(());
    }

    let bind = if config.public_api && config.lan {
        Ipv4Addr::UNSPECIFIED
    } else {
        Ipv4Addr::LOCALHOST
    };
    let port = if config.public_api { config.port } else { 0 };
    ensure!(
        !config.public_api || port != 0,
        "Укажи порт API от 1 до 65535."
    );
    let port = TcpListener::bind((bind, port))
        .context("Порт уже занят или недоступен")?
        .local_addr()?
        .port();
    let mut endpoint = Endpoint {
        port,
        key: if config.public_api {
            config.api_key.clone()
        } else {
            Uuid::new_v4().simple().to_string()
        },
        backend,
        exposed: config.public_api,
        lan: config.public_api && config.lan,
        context: config.context,
    };
    let mut command = Command::new(binary);
    command.args(server_args(&config, &endpoint, bind)?);
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("LLAMA_ARG_") {
            command.env_remove(name);
        }
    }
    command
        .env_remove("LLAMA_API_KEY")
        .env_remove("DORY_WORKBENCH_TRACE");
    if config.trace {
        command.env("DORY_WORKBENCH_TRACE", "1");
    }
    tell(EngineEvent::Stage("Загрузка модели".into()));
    let mut server =
        Process::start(command, tell.clone()).context("Не удалось запустить llama-server")?;
    let start = Instant::now();
    let mut ready = false;
    while !stop.load(Ordering::Relaxed) {
        if let Some(status) = server.0.try_wait()? {
            bail!("llama-server завершился ({status}). Подробности в журнале.");
        }
        if !ready && let Some(context) = ready_context(&endpoint) {
            ready = true;
            endpoint.context = context;
            tell(EngineEvent::Ready(endpoint.clone()));
        }
        if !ready && start.elapsed() > Duration::from_secs(300) {
            bail!("Модель не загрузилась за 5 минут. Подробности в журнале.");
        }
        thread::sleep(Duration::from_millis(120));
    }
    Ok(())
}

fn server_args(
    config: &EngineConfig,
    endpoint: &Endpoint,
    bind: Ipv4Addr,
) -> Result<Vec<OsString>> {
    let mut args: Vec<OsString> = Vec::new();
    args.extend([
        "--model".into(),
        config.model.canonicalize()?.into_os_string(),
        "--host".into(),
        bind.to_string().into(),
        "--port".into(),
        endpoint.port.to_string().into(),
        "--api-key".into(),
        endpoint.key.clone().into(),
        "--alias".into(),
        MODEL_ID.into(),
    ]);
    let layers = if endpoint.backend == "CPU" {
        "0"
    } else {
        &config.gpu_layers
    };
    for (key, value) in [
        ("--ctx-size", config.context.to_string()),
        ("--batch-size", config.batch.to_string()),
        ("--ubatch-size", config.ubatch.to_string()),
        ("--gpu-layers", layers.into()),
        ("--flash-attn", config.flash.clone()),
        ("--cache-type-k", config.key_cache.clone()),
        ("--cache-type-v", config.value_cache.clone()),
        ("--fit-target", config.fit_margin.to_string()),
        ("--parallel", "1".into()),
    ] {
        args.extend([key.into(), value.into()]);
    }
    if config.threads > 0 {
        args.extend(["--threads".into(), config.threads.to_string().into()]);
    }
    if config.batch_threads > 0 {
        args.extend([
            "--threads-batch".into(),
            config.batch_threads.to_string().into(),
        ]);
    }
    if !config.warmup {
        args.push("--no-warmup".into());
    }
    args.push(
        if config.context_shift {
            "--context-shift"
        } else {
            "--no-context-shift"
        }
        .into(),
    );
    args.extend([
        "--no-ui".into(),
        "--jinja".into(),
        "--reasoning-format".into(),
        "deepseek".into(),
    ]);
    let mut overrides = Vec::new();
    if config.dory_loops > 0 {
        overrides.push(format!("dory.num_loops=int:{}", config.dory_loops));
    }
    if config.dory_kv != "model" {
        overrides.push(format!(
            "dory.recurrent_kv_cache_mode=str:{}",
            config.dory_kv
        ));
    }
    if !overrides.is_empty() {
        args.extend(["--override-kv".into(), overrides.join(",").into()]);
    }
    if !config.template.as_os_str().is_empty() {
        args.extend([
            "--chat-template-file".into(),
            config.template.canonicalize()?.into_os_string(),
        ]);
    }
    Ok(args)
}

fn engine_binary(
    config: &EngineConfig,
    stop: &AtomicBool,
    tell: &(impl Fn(EngineEvent) + Clone + Send + 'static),
) -> Result<(PathBuf, String)> {
    if config.backend == Backend::Existing {
        ensure!(config.executable.is_file(), "Выбери файл llama-server.");
        return Ok((config.executable.canonicalize()?, "Свой сервер".into()));
    }
    let source = config
        .source
        .canonicalize()
        .context("Укажи папку с форком llama.cpp")?;
    ensure!(
        source.join("src/models/dory.cpp").is_file(),
        "В папке отсутствует реализация Dory."
    );
    let cuda = match config.backend {
        Backend::Cuda => true,
        Backend::Auto => locate_nvcc().is_some() && nvidia_device(),
        _ => false,
    };
    let result = build_backend(config, &source, cuda, stop, tell);
    match result {
        Err(error) if cuda && config.backend == Backend::Auto && !stop.load(Ordering::Relaxed) => {
            tell(EngineEvent::Log(format!(
                "Сборка CUDA недоступна: {error:#}. Переход на CPU."
            )));
            build_backend(config, &source, false, stop, tell)
        }
        other => other,
    }
}

fn build_backend(
    config: &EngineConfig,
    source: &Path,
    cuda: bool,
    stop: &AtomicBool,
    tell: &(impl Fn(EngineEvent) + Clone + Send + 'static),
) -> Result<(PathBuf, String)> {
    let backend = if cuda { "CUDA" } else { "CPU" };
    let dir = source
        .join("apps/dory-workbench/.engine-build")
        .join(backend.to_ascii_lowercase());
    fs::create_dir_all(&dir)?;
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join("build.lock"))?;
    lock.try_lock()
        .context("Другой экземпляр уже собирает этот движок")?;
    tell(EngineEvent::Stage(format!("Подготовка движка {backend}")));
    let mut configure = Command::new(&config.cmake);
    configure.arg("-S").arg(source).arg("-B").arg(&dir).args([
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_UI=OFF",
        "-DLLAMA_USE_PREBUILT_UI=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_OPENSSL=OFF",
        if cuda {
            "-DGGML_CUDA=ON"
        } else {
            "-DGGML_CUDA=OFF"
        },
    ]);
    if cuda {
        let compiler = locate_nvcc().context("Для CUDA-сборки нужен CUDA Toolkit с nvcc")?;
        configure.arg(format!("-DCMAKE_CUDA_COMPILER={}", compiler.display()));
        if let Some(architectures) = cuda_architectures() {
            configure.arg(format!("-DCMAKE_CUDA_ARCHITECTURES={architectures}"));
        }
    }
    build_step(configure, stop, tell).context("Не удалось настроить сборку CMake")?;
    let mut build = Command::new(&config.cmake);
    let jobs = thread::available_parallelism().map_or(2, |n| n.get().min(6));
    build.arg("--build").arg(&dir).args([
        "--config",
        "Release",
        "--target",
        "llama-server",
        "--parallel",
        &jobs.to_string(),
    ]);
    build_step(build, stop, tell).context("Не удалось собрать llama-server")?;
    let name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let binary = [
        dir.join("bin/Release").join(name),
        dir.join("bin").join(name),
    ]
    .into_iter()
    .find(|p| p.is_file())
    .context("CMake не создал llama-server")?;
    Ok((binary, backend.into()))
}

fn locate_nvcc() -> Option<PathBuf> {
    let name = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    let mut paths = Vec::new();
    if let Some(cuda) = std::env::var_os("CUDA_PATH") {
        paths.push(PathBuf::from(cuda).join("bin"));
    }
    if let Some(path) = std::env::var_os("PATH") {
        paths.extend(std::env::split_paths(&path));
    }
    paths.push(PathBuf::from("/usr/local/cuda/bin"));
    paths
        .into_iter()
        .map(|p| p.join(name))
        .find(|p| p.is_file())
}

fn nvidia_device() -> bool {
    let mut command = Command::new("nvidia-smi");
    quiet(&mut command);
    command
        .arg("-L")
        .output()
        .is_ok_and(|out| out.status.success() && !out.stdout.is_empty())
}

fn cuda_architectures() -> Option<String> {
    let mut command = Command::new("nvidia-smi");
    quiet(&mut command);
    let out = command
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let mut arches: Vec<_> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| {
            let (major, minor) = line.trim().split_once('.')?;
            let major = major.parse::<u32>().ok()?;
            let minor = minor.parse::<u32>().ok()?;
            if major >= 12 || minor > 9 {
                return None;
            }
            Some(format!("{major}{minor}-real"))
        })
        .collect();
    arches.sort();
    arches.dedup();
    (!arches.is_empty()).then(|| arches.join(";"))
}

fn ready_context(endpoint: &Endpoint) -> Option<u32> {
    let timeout = Duration::from_millis(300);
    let address = (Ipv4Addr::LOCALHOST, endpoint.port).into();
    let mut stream = TcpStream::connect_timeout(&address, timeout).ok()?;
    stream.set_read_timeout(Some(timeout)).ok()?;
    stream.set_write_timeout(Some(timeout)).ok()?;
    use std::io::Write;
    write!(stream, "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {}\r\nConnection: close\r\n\r\n", endpoint.key).ok()?;
    let mut response = String::new();
    stream.take(64 * 1024).read_to_string(&mut response).ok()?;
    let (header, body) = response.split_once("\r\n\r\n")?;
    if !header.starts_with("HTTP/1.1 200 ") {
        return None;
    }
    let body = serde_json::from_str::<Value>(body).ok()?;
    let model = body["data"]
        .as_array()?
        .iter()
        .find(|m| m["id"] == MODEL_ID)?;
    let limit = model["meta"]["n_ctx"]
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .unwrap_or(endpoint.context);
    Some(limit.min(endpoint.context))
}

fn build_step(
    command: Command,
    stop: &AtomicBool,
    tell: &(impl Fn(EngineEvent) + Clone + Send + 'static),
) -> Result<()> {
    ensure!(!stop.load(Ordering::Relaxed), "Сборка прервана");
    let mut child = Process::start(command, tell.clone())?;
    loop {
        ensure!(!stop.load(Ordering::Relaxed), "Сборка прервана");
        if let Some(status) = child.0.try_wait()? {
            ensure!(
                status.success(),
                "Команда завершилась ({status}). Подробности в журнале."
            );
            return Ok(());
        }
        thread::sleep(Duration::from_millis(120));
    }
}

fn quiet(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    #[cfg(not(windows))]
    let _ = command;
}

struct Process(Child);

impl Process {
    fn start(
        mut command: Command,
        tell: impl Fn(EngineEvent) + Clone + Send + 'static,
    ) -> Result<Self> {
        let mut child = spawn_group(
            command
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped()),
        )?;
        if let Some(out) = child.stdout.take() {
            read_logs(out, tell.clone());
        }
        if let Some(err) = child.stderr.take() {
            read_logs(err, tell);
        }
        Ok(Self(child))
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        if matches!(self.0.try_wait(), Ok(Some(_))) {
            return;
        }
        kill_group(&mut self.0);
    }
}

pub(crate) fn spawn_group(command: &mut Command) -> std::io::Result<Child> {
    quiet(command);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    command.spawn()
}

pub(crate) fn kill_group(child: &mut Child) {
    #[cfg(unix)]
    unsafe {
        libc::kill(-(child.id() as i32), libc::SIGKILL);
    }
    #[cfg(windows)]
    {
        let mut taskkill = Command::new("taskkill");
        quiet(&mut taskkill);
        let _ = taskkill
            .args(["/T", "/F", "/PID", &child.id().to_string()])
            .output();
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn read_logs(reader: impl Read + Send + 'static, tell: impl Fn(EngineEvent) + Send + 'static) {
    thread::spawn(move || {
        let reader = BufReader::new(reader);
        for mut line in reader.lines().map_while(Result::ok) {
            if line.len() > 2000 {
                line.truncate(line.floor_char_boundary(2000));
            }
            if let Some((round, total)) = line
                .strip_prefix("DORY_CYCLE ")
                .and_then(|text| text.split_once('/'))
                .and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))
            {
                tell(EngineEvent::Cycle(round, total));
            } else {
                tell(EngineEvent::Log(line));
            }
        }
    });
}
