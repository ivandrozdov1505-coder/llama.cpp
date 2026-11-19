use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backend {
    #[default]
    Auto,
    Cuda,
    Cpu,
    Existing,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Авто",
            Self::Cuda => "CUDA",
            Self::Cpu => "CPU",
            Self::Existing => "Готовый сервер",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct EngineConfig {
    pub model: PathBuf,
    pub models_dir: PathBuf,
    pub source: PathBuf,
    pub executable: PathBuf,
    pub cmake: String,
    pub backend: Backend,
    pub context: u32,
    pub batch: u32,
    pub ubatch: u32,
    pub threads: u32,
    pub batch_threads: u32,
    pub gpu_layers: String,
    pub flash: String,
    pub key_cache: String,
    pub value_cache: String,
    pub fit_margin: u32,
    pub context_shift: bool,
    pub warmup: bool,
    pub dory_loops: u32,
    pub dory_kv: String,
    pub template: PathBuf,
    pub trace: bool,
    pub api_key: String,
    pub public_api: bool,
    pub lan: bool,
    pub port: u16,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            models_dir: PathBuf::new(),
            source: source_root(),
            executable: PathBuf::new(),
            cmake: "cmake".into(),
            backend: Backend::Auto,
            context: 4096,
            batch: 512,
            ubatch: 256,
            threads: 0,
            batch_threads: 0,
            gpu_layers: "auto".into(),
            flash: "auto".into(),
            key_cache: "f16".into(),
            value_cache: "f16".into(),
            fit_margin: 512,
            context_shift: true,
            warmup: true,
            dory_loops: 0,
            dory_kv: "model".into(),
            template: PathBuf::new(),
            trace: false,
            api_key: uuid::Uuid::new_v4().simple().to_string(),
            public_api: false,
            lan: false,
            port: 8080,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Sampling {
    pub system: String,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub tokens: u32,
    pub seed: i32,
    pub reasoning: String,
    pub reasoning_budget: i32,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            system: String::new(),
            temperature: 0.7,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            repeat_penalty: 1.05,
            tokens: 1024,
            seed: -1,
            reasoning: "auto".into(),
            reasoning_budget: -1,
        }
    }
}

pub fn is_model(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        && path.is_file()
}

fn source_root() -> PathBuf {
    let roots = [
        std::env::current_dir().ok(),
        std::env::current_exe().ok(),
        Some(PathBuf::from(env!("CARGO_MANIFEST_DIR"))),
    ];
    roots
        .into_iter()
        .flatten()
        .find_map(|path| {
            path.ancestors()
                .find(|p| {
                    p.join("src/models/dory.cpp").is_file() && p.join("CMakeLists.txt").is_file()
                })
                .map(Path::to_path_buf)
        })
        .unwrap_or_default()
}
