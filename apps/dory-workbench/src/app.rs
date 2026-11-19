use crate::{
    chat::{ChatEvent, Conversation, Message, Turn},
    config::{Backend, EngineConfig, Sampling, is_model},
    engine::{Endpoint, Engine, EngineEvent, MODEL_ID},
    tools::Action,
};
use eframe::egui::{
    self, Align, Color32, Context, Frame, Layout, Margin, RichText, ScrollArea, Stroke, TextEdit,
    TextStyle, Ui, Vec2,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
    },
};
use tokio::{runtime::Runtime, sync::oneshot, task::JoinHandle};

const BG: Color32 = Color32::from_rgb(14, 19, 26);
const PANEL: Color32 = Color32::from_rgb(19, 27, 36);
const CARD: Color32 = Color32::from_rgb(24, 35, 46);
const LINE: Color32 = Color32::from_rgb(46, 62, 76);
const TEXT: Color32 = Color32::from_rgb(229, 238, 246);
const MUTE: Color32 = Color32::from_rgb(154, 176, 194);
const ACCENT: Color32 = Color32::from_rgb(94, 226, 192);
const INK: Color32 = Color32::from_rgb(11, 30, 29);
const WARN: Color32 = Color32::from_rgb(255, 196, 128);

#[derive(Serialize, Deserialize)]
#[serde(default)]
struct Saved {
    engine: EngineConfig,
    sampling: Sampling,
    conversations: Vec<Conversation>,
    selected: usize,
    agent: bool,
    workspace: PathBuf,
}

impl Default for Saved {
    fn default() -> Self {
        Self {
            engine: EngineConfig::default(),
            sampling: Sampling::default(),
            conversations: vec![new_conversation()],
            selected: 0,
            agent: false,
            workspace: PathBuf::new(),
        }
    }
}

fn new_conversation() -> Conversation {
    Conversation {
        title: "Новый диалог".into(),
        ..Default::default()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Model,
    Sampling,
    Api,
}

enum Event {
    Engine(EngineEvent),
    Chat(u64, ChatEvent),
}

struct Generation {
    id: u64,
    task: JoinHandle<()>,
    stop: Arc<AtomicBool>,
}

impl Drop for Generation {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.task.abort();
    }
}

type Approval = (Action, oneshot::Sender<bool>);

pub struct Workbench {
    saved: Saved,
    runtime: Runtime,
    sender: Sender<Event>,
    events: Receiver<Event>,
    engine: Option<Engine>,
    endpoint: Option<Endpoint>,
    generation: Option<Generation>,
    next_request: u64,
    approval: Option<Approval>,
    models: Vec<PathBuf>,
    logs: VecDeque<String>,
    show_logs: bool,
    show_about_chat: bool,
    tab: Tab,
    draft: String,
    reply: String,
    reasoning: String,
    error: String,
    engine_status: String,
    activity: String,
    cycle: Option<(u32, u32)>,
    ctx_tokens: Option<usize>,
    feedback: String,
    speed: Option<f64>,
}

impl Workbench {
    pub fn new(cc: &eframe::CreationContext<'_>) -> anyhow::Result<Self> {
        style(&cc.egui_ctx);
        let mut saved: Saved = cc
            .storage
            .and_then(|s| eframe::get_value(s, "workbench-v1"))
            .unwrap_or_default();
        if saved.conversations.is_empty() {
            saved.conversations.push(new_conversation());
        }
        saved.selected = saved.selected.min(saved.conversations.len() - 1);
        if saved.engine.api_key.is_empty() {
            saved.engine.api_key = uuid::Uuid::new_v4().simple().to_string();
        }
        let models = scan_models(&saved.engine.models_dir);
        let (sender, events) = mpsc::channel();
        Ok(Self {
            saved,
            runtime: Runtime::new()?,
            sender,
            events,
            models,
            engine: None,
            endpoint: None,
            generation: None,
            approval: None,
            next_request: 0,
            logs: VecDeque::new(),
            show_logs: false,
            show_about_chat: false,
            tab: Tab::Model,
            draft: String::new(),
            reply: String::new(),
            reasoning: String::new(),
            error: String::new(),
            engine_status: "Движок выключен".into(),
            activity: String::new(),
            cycle: None,
            ctx_tokens: None,
            feedback: String::new(),
            speed: None,
        })
    }

    fn read_events(&mut self) {
        while let Ok(event) = self.events.try_recv() {
            match event {
                Event::Engine(event) => self.engine_event(event),
                Event::Chat(id, event) if self.generation.as_ref().is_some_and(|g| g.id == id) => {
                    self.chat_event(event)
                }
                Event::Chat(_, _) => {}
            }
        }
    }

    fn engine_event(&mut self, event: EngineEvent) {
        match event {
            EngineEvent::Stage(status) => self.engine_status = status,
            EngineEvent::Log(line) => {
                if self.logs.len() >= 600 {
                    self.logs.pop_front();
                }
                self.logs.push_back(line);
            }
            EngineEvent::Ready(endpoint) => {
                self.engine_status = format!("Готово · {}", endpoint.backend);
                self.endpoint = Some(endpoint);
                self.cycle = None;
            }
            EngineEvent::Cycle(round, total)
                if self.endpoint.is_some() && total > 0 && round <= total =>
            {
                self.cycle = Some((round, total));
            }
            EngineEvent::Cycle(_, _) => {}
            EngineEvent::Exited(result) => {
                self.engine.take();
                self.endpoint = None;
                self.stop_chat();
                self.engine_status = "Движок выключен".into();
                if let Err(error) = result {
                    self.error = error;
                }
            }
        }
    }

    fn chat_event(&mut self, event: ChatEvent) {
        match event {
            ChatEvent::Stage(text) => self.activity = text,
            ChatEvent::Delta { answer, reasoning } => {
                self.reply.push_str(&answer);
                self.reasoning.push_str(&reasoning);
            }
            ChatEvent::Messages(messages) => {
                self.saved.conversations[self.saved.selected]
                    .history
                    .extend(messages);
                self.reply.clear();
                self.reasoning.clear();
            }
            ChatEvent::Memory(memory, tokens) => {
                self.saved.conversations[self.saved.selected].memory = memory;
                self.ctx_tokens = Some(tokens);
                self.feedback = "Контекст сжат. Полная история сохранена.".into();
            }
            ChatEvent::Tokens(tokens) => self.ctx_tokens = Some(tokens),
            ChatEvent::Speed(speed) => self.speed = Some(speed),
            ChatEvent::Approval(action, answer) => {
                self.activity = "Ожидает подтверждения".into();
                self.approval = Some((action, answer));
            }
            ChatEvent::Finished(result) => {
                self.generation.take();
                self.approval.take();
                self.activity.clear();
                if let Err(error) = result {
                    self.error = error;
                    self.keep_partial();
                }
            }
        }
    }

    fn start_engine(&mut self, ctx: &Context) {
        self.error.clear();
        self.feedback.clear();
        self.logs.clear();
        self.ctx_tokens = None;
        self.cycle = None;
        self.engine_status = "Запуск движка".into();
        let sender = self.sender.clone();
        let ctx = ctx.clone();
        self.engine = Some(Engine::start(self.saved.engine.clone(), move |event| {
            let _ = sender.send(Event::Engine(event));
            ctx.request_repaint();
        }));
    }

    fn stop_engine(&mut self) {
        self.stop_chat();
        self.endpoint = None;
        if let Some(engine) = &self.engine {
            engine.stop();
            self.engine_status = "Остановка движка".into();
        }
    }

    fn start_chat(&mut self, ctx: &Context) {
        let Some(endpoint) = self.endpoint.clone() else {
            return;
        };
        if self.generation.is_some() || self.draft.trim().is_empty() {
            return;
        }
        let workspace = if self.saved.agent {
            match self.saved.workspace.canonicalize() {
                Ok(path) if path.is_dir() => Some(path),
                _ => {
                    self.error = "Выбери рабочую папку для агента.".into();
                    return;
                }
            }
        } else {
            None
        };
        self.error.clear();
        self.feedback.clear();
        self.cycle = None;
        self.speed = None;
        let input = std::mem::take(&mut self.draft);
        let conversation = &mut self.saved.conversations[self.saved.selected];
        if conversation.history.is_empty() {
            conversation.title = input.trim().chars().take(36).collect();
        }
        conversation
            .history
            .push(Message::new("user", input.trim()));
        let stop = Arc::new(AtomicBool::new(false));
        let turn = Turn {
            endpoint,
            conversation: conversation.clone(),
            sampling: self.saved.sampling.clone(),
            workspace,
            stop: stop.clone(),
        };
        self.next_request += 1;
        let id = self.next_request;
        let sender = self.sender.clone();
        let ctx = ctx.clone();
        let task = self.runtime.spawn(async move {
            let emit = |event| {
                let _ = sender.send(Event::Chat(id, event));
                ctx.request_repaint();
            };
            let result = turn.run(&emit).await.map_err(|error| format!("{error:#}"));
            emit(ChatEvent::Finished(result));
        });
        self.generation = Some(Generation { id, task, stop });
        self.activity = "Подготовка запроса".into();
    }

    fn keep_partial(&mut self) {
        if !self.reply.is_empty() {
            let mut message = Message::new("assistant", std::mem::take(&mut self.reply));
            message.reasoning_content = std::mem::take(&mut self.reasoning);
            self.saved.conversations[self.saved.selected]
                .history
                .push(message);
        }
        self.reasoning.clear();
    }

    fn stop_chat(&mut self) {
        self.generation.take();
        self.approval.take();
        self.activity.clear();
        self.keep_partial();
    }

    fn choose_model(&mut self, path: PathBuf) {
        if is_model(&path) {
            self.saved.engine.model = path;
            self.error.clear();
        } else {
            self.error = "Выбери существующий файл .gguf.".into();
        }
    }

    fn sidebar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            Frame::new()
                .fill(ACCENT)
                .corner_radius(8)
                .inner_margin(Margin::symmetric(11, 7))
                .show(ui, |ui| {
                    ui.label(RichText::new("D").strong().color(INK).size(21.0));
                });
            ui.vertical(|ui| {
                ui.label(RichText::new("Dory").size(24.0).strong());
                ui.label(RichText::new("WORKBENCH").small().color(MUTE));
            });
        });
        ui.add_space(22.0);
        heading(ui, "Локальная модель");
        let enabled = self.engine.is_none();
        ui.add_enabled_ui(enabled, |ui| {
            if ui.button("Выбрать GGUF…").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("GGUF", &["gguf"])
                    .pick_file()
            {
                self.choose_model(path);
            }
            if ui.button("Папка моделей…").clicked()
                && let Some(path) = rfd::FileDialog::new().pick_folder()
            {
                self.saved.engine.models_dir = path;
                self.models = scan_models(&self.saved.engine.models_dir);
            }
            if !self.saved.engine.models_dir.as_os_str().is_empty() {
                ui.add_space(4.0);
                ui.label(
                    RichText::new(self.saved.engine.models_dir.display().to_string())
                        .small()
                        .color(MUTE),
                );
            }
            if !self.models.is_empty() {
                ScrollArea::vertical()
                    .id_salt("models")
                    .max_height(132.0)
                    .show(ui, |ui| {
                        let mut picked = None;
                        for model in &self.models {
                            let selected = self.saved.engine.model == *model;
                            if ui
                                .selectable_label(selected, name(model))
                                .on_hover_text(model.display().to_string())
                                .clicked()
                            {
                                picked = Some(model.clone());
                            }
                        }
                        if let Some(model) = picked {
                            self.choose_model(model);
                        }
                    });
            }
        });
        ui.add_space(8.0);
        card(ui, |ui| {
            ui.label(RichText::new(name(&self.saved.engine.model)).strong());
            ui.label(RichText::new(self.engine_status.as_str()).color(
                if self.endpoint.is_some() {
                    ACCENT
                } else {
                    MUTE
                },
            ));
            if self.engine.is_some() {
                if ui.button("Остановить движок").clicked() {
                    self.stop_engine();
                }
            } else if ui
                .add_enabled(is_model(&self.saved.engine.model), primary("Запустить"))
                .clicked()
            {
                self.start_engine(ui.ctx());
            }
        });
        ui.add_space(20.0);
        ui.horizontal(|ui| {
            heading(ui, "Диалоги");
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui
                    .add_enabled(self.generation.is_none(), egui::Button::new("+ Новый"))
                    .clicked()
                {
                    self.saved.conversations.push(new_conversation());
                    self.saved.selected = self.saved.conversations.len() - 1;
                    self.ctx_tokens = None;
                }
            });
        });
        ScrollArea::vertical()
            .id_salt("chats")
            .max_height((ui.available_height() - 48.0).max(60.0))
            .show(ui, |ui| {
                ui.add_enabled_ui(self.generation.is_none(), |ui| {
                    for (id, conversation) in self.saved.conversations.iter().enumerate().rev() {
                        if ui
                            .selectable_label(id == self.saved.selected, &conversation.title)
                            .clicked()
                        {
                            self.saved.selected = id;
                            self.ctx_tokens = None;
                        }
                    }
                });
            });
        ui.with_layout(Layout::bottom_up(Align::Min), |ui| {
            if ui.button("Журнал движка").clicked() {
                self.show_logs = true;
            }
        });
    }

    fn settings(&mut self, ui: &mut Ui) {
        heading(ui, "Настройки");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tab, Tab::Model, "Модель");
            ui.selectable_value(&mut self.tab, Tab::Sampling, "Ответ");
            ui.selectable_value(&mut self.tab, Tab::Api, "API");
        });
        ui.add_space(8.0);
        ScrollArea::vertical()
            .id_salt("configuration")
            .show(ui, |ui| match self.tab {
                Tab::Model => {
                    if self.engine.is_some() {
                        ui.label(
                            RichText::new("Останови движок для изменения.")
                                .small()
                                .color(MUTE),
                        );
                    }
                    ui.add_enabled_ui(self.engine.is_none(), |ui| {
                        model_settings(ui, &mut self.saved.engine)
                    });
                }
                Tab::Sampling => {
                    ui.add_enabled_ui(self.generation.is_none(), |ui| {
                        sampling_settings(ui, &mut self.saved.sampling)
                    });
                    ui.add_space(12.0);
                    if let Some(endpoint) = &self.endpoint {
                        token_meter(ui, self.ctx_tokens, endpoint.context);
                    }
                }
                Tab::Api => self.api_settings(ui),
            });
    }

    fn api_settings(&mut self, ui: &mut Ui) {
        section(ui, "Подключение клиентов");
        ui.add_enabled_ui(self.engine.is_none(), |ui| {
            ui.checkbox(
                &mut self.saved.engine.public_api,
                "Включить API для клиентов",
            );
            ui.add_enabled_ui(self.saved.engine.public_api, |ui| {
                number(ui, "Порт", &mut self.saved.engine.port, 1..=65535);
                ui.checkbox(
                    &mut self.saved.engine.lan,
                    "Доступ с устройств в локальной сети",
                );
                if self.saved.engine.lan {
                    ui.label(
                        RichText::new("HTTP без шифрования. Используй в доверенной сети.")
                            .color(WARN)
                            .small(),
                    );
                }
            });
        });
        let Some(endpoint) = self.endpoint.as_ref().filter(|endpoint| endpoint.exposed) else {
            ui.add_space(6.0);
            ui.label(
                RichText::new("Включи API и запусти модель, чтобы получить адрес и ключ.")
                    .color(MUTE),
            );
            return;
        };
        ui.add_space(10.0);
        card(ui, |ui| {
            ui.label(RichText::new("OpenAI-compatible").strong().color(ACCENT));
            ui.monospace(endpoint.url());
            ui.label(format!("Модель: {MODEL_ID}"));
            if endpoint.lan {
                ui.label(
                    RichText::new("Для другого устройства подставь IP этого ПК.")
                        .color(MUTE)
                        .small(),
                );
            }
        });
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button("Копировать URL").clicked() {
                ui.ctx().copy_text(endpoint.url());
            }
            if ui.button("Копировать ключ").clicked() {
                ui.ctx().copy_text(endpoint.key.clone());
            }
        });
        section(ui, "OpenCode");
        ui.label(
            RichText::new("Вставь JSON в opencode.json в папке проекта.")
                .color(MUTE)
                .small(),
        );
        if ui.button("Копировать конфигурацию").clicked() {
            ui.ctx().copy_text(endpoint.opencode_config(false));
        }
        if ui.button("Конфигурация для OpenCode v1").clicked() {
            ui.ctx().copy_text(endpoint.opencode_config(true));
        }
    }

    fn central(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(
                    RichText::new(if self.saved.agent {
                        "Рабочая среда"
                    } else {
                        "Чат с моделью"
                    })
                    .size(28.0)
                    .strong(),
                );
                ui.label(
                    RichText::new(if self.saved.agent {
                        "Файлы · Поиск · PowerShell"
                    } else {
                        "Локальная генерация"
                    })
                    .color(MUTE),
                );
            });
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.add_enabled_ui(self.generation.is_none(), |ui| {
                    ui.selectable_value(&mut self.saved.agent, true, "Агент");
                    ui.selectable_value(&mut self.saved.agent, false, "Чат");
                });
            });
        });
        if self.saved.agent {
            ui.add_space(8.0);
            card(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add_enabled(
                            self.generation.is_none(),
                            egui::Button::new("Рабочая папка…"),
                        )
                        .clicked()
                        && let Some(path) = rfd::FileDialog::new().pick_folder()
                    {
                        self.saved.workspace = path;
                    }
                    if !self.saved.workspace.as_os_str().is_empty() {
                        ui.label(
                            RichText::new(self.saved.workspace.display().to_string())
                                .color(MUTE)
                                .small(),
                        );
                    }
                });
            });
        }

        ui.add_space(10.0);
        if self.saved.engine.trace {
            self.rounds(ui);
            ui.add_space(8.0);
        }
        egui::TopBottomPanel::bottom("composer")
            .frame(Frame::new().fill(BG))
            .show_inside(ui, |ui| self.composer(ui));
        let height = ui.available_height().max(30.0);
        let conversation = &self.saved.conversations[self.saved.selected];
        let history = &conversation.history;
        ScrollArea::vertical()
            .id_salt(("transcript", self.saved.selected))
            .max_height(height)
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                if history.is_empty() {
                    ui.add_space(28.0);
                    card(ui, |ui| {
                        ui.label(
                            RichText::new(if self.saved.agent {
                                "Открой проект и поставь задачу"
                            } else {
                                "Начни диалог"
                            })
                            .size(22.0)
                            .strong(),
                        );
                        ui.label(
                            RichText::new(if self.endpoint.is_some() {
                                "Напиши сообщение в поле ниже."
                            } else {
                                "Выбери GGUF слева и запусти движок."
                            })
                            .color(MUTE),
                        );
                    });
                }
                for (index, message) in history.iter().enumerate() {
                    ui.push_id(index, |ui| message_card(ui, message));
                }
                if !self.reply.is_empty() || !self.reasoning.is_empty() {
                    let mut message = Message::new("assistant", &self.reply);
                    message.reasoning_content.clone_from(&self.reasoning);
                    message_card(ui, &message);
                }
            });
    }

    fn composer(&mut self, ui: &mut Ui) {
        ui.add_space(8.0);
        self.approval_card(ui);
        if !self.error.is_empty() {
            ScrollArea::vertical()
                .id_salt("errors")
                .max_height(45.0)
                .show(ui, |ui| {
                    ui.label(RichText::new(&self.error).color(WARN));
                });
        }
        if !self.activity.is_empty() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new(&self.activity).color(MUTE));
            });
        } else if !self.feedback.is_empty() {
            ui.label(RichText::new(&self.feedback).small().color(ACCENT));
        }
        ui.add_space(6.0);
        card(ui, |ui| {
            let prompt = ui.add_sized(
                [ui.available_width(), 64.0],
                TextEdit::multiline(&mut self.draft)
                    .hint_text(if self.saved.agent {
                        "Опиши задачу для агента…"
                    } else {
                        "Сообщение модели…"
                    })
                    .frame(false),
            );
            let shortcut = prompt.has_focus()
                && ui.input(|input| input.modifiers.command && input.key_pressed(egui::Key::Enter));
            ui.horizontal(|ui| {
                if self.generation.is_some() {
                    if ui.button("Прервать").clicked() {
                        self.stop_chat();
                    }
                } else {
                    let enabled = self.endpoint.is_some() && !self.draft.trim().is_empty();
                    let send = ui.add_enabled(enabled, primary("Отправить")).clicked();
                    if send || (enabled && shortcut) {
                        self.start_chat(ui.ctx());
                    }
                }
                ui.label(RichText::new("Ctrl + Enter").small().color(MUTE));
                if let Some(speed) = self.speed {
                    ui.label(
                        RichText::new(format!("{speed:.1} ток/с"))
                            .small()
                            .color(ACCENT),
                    );
                }
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.button("Контекст").clicked() {
                        self.show_about_chat = true;
                    }
                });
            });
        });
    }

    fn rounds(&self, ui: &mut Ui) {
        let (current, total) = self.cycle.unwrap_or((0, self.saved.engine.dory_loops));
        ui.horizontal(|ui| {
            ui.label(RichText::new("Круги Dory").strong());
            if total == 0 {
                ui.label(RichText::new("Ждём вычислений").color(MUTE).small());
            }
            for round in 1..=total.min(4) {
                let on = round <= current;
                Frame::new()
                    .fill(if on { ACCENT } else { CARD })
                    .corner_radius(5)
                    .inner_margin(Margin::symmetric(8, 3))
                    .show(ui, |ui| {
                        ui.label(RichText::new(round.to_string()).color(if on {
                            INK
                        } else {
                            MUTE
                        }));
                    });
            }
            if current > 0 {
                ui.label(
                    RichText::new(format!(
                        "{current} / {total} · последний завершённый проход"
                    ))
                    .small()
                    .color(MUTE),
                );
            }
        });
    }

    fn approval_card(&mut self, ui: &mut Ui) {
        let Some((action, _)) = self.approval.as_ref() else {
            return;
        };
        let choice = card(ui, |ui| {
            ui.label(RichText::new(action.title()).color(WARN).strong());
            if matches!(action, Action::PowerShell(_)) {
                ui.label(
                    RichText::new("Команда запускается с правами пользователя.")
                        .small()
                        .color(MUTE),
                );
            }
            ScrollArea::vertical()
                .id_salt("approval")
                .max_height(88.0)
                .show(ui, |ui| {
                    let mut view = action.preview();
                    ui.add(
                        TextEdit::multiline(&mut view)
                            .font(TextStyle::Monospace)
                            .desired_width(f32::INFINITY),
                    );
                });
            let mut value = None;
            ui.horizontal(|ui| {
                if ui.add(primary("Разрешить один раз")).clicked() {
                    value = Some(true);
                }
                if ui.button("Отклонить").clicked() {
                    value = Some(false);
                }
            });
            value
        });
        if let Some(choice) = choice
            && let Some((_, sender)) = self.approval.take()
        {
            let _ = sender.send(choice);
        }
        ui.add_space(8.0);
    }
}

impl eframe::App for Workbench {
    fn update(&mut self, ctx: &Context, _: &mut eframe::Frame) {
        self.read_events();
        if self.engine.is_none()
            && let Some(path) = ctx.input(|i| {
                i.raw
                    .dropped_files
                    .iter()
                    .find_map(|file| file.path.clone())
            })
        {
            self.choose_model(path);
        }
        egui::SidePanel::left("library")
            .resizable(true)
            .default_width(245.0)
            .width_range(220.0..=325.0)
            .frame(Frame::new().fill(PANEL).inner_margin(16))
            .show(ctx, |ui| self.sidebar(ui));
        egui::SidePanel::right("parameters")
            .resizable(true)
            .default_width(310.0)
            .width_range(280.0..=450.0)
            .frame(Frame::new().fill(PANEL).inner_margin(16))
            .show(ctx, |ui| self.settings(ui));
        egui::CentralPanel::default()
            .frame(Frame::new().fill(BG).inner_margin(18))
            .show(ctx, |ui| self.central(ui));

        egui::Window::new("Журнал движка")
            .open(&mut self.show_logs)
            .default_width(690.0)
            .show(ctx, |ui| {
                if ui.button("Копировать журнал").clicked() {
                    ctx.copy_text(self.logs.iter().cloned().collect::<Vec<_>>().join("\n"));
                }
                ScrollArea::vertical()
                    .max_height(450.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in &self.logs {
                            ui.monospace(line);
                        }
                    });
            });
        egui::Window::new("Окно контекста")
            .open(&mut self.show_about_chat)
            .show(ctx, |ui| {
                if let Some(endpoint) = &self.endpoint {
                    token_meter(ui, self.ctx_tokens, endpoint.context);
                }
                let memory = &self.saved.conversations[self.saved.selected].memory;
                ui.label("Сжатие запускается автоматически при заполнении окна.");
                if memory.through > 0 {
                    ui.label(format!("Сообщений в резюме: {}", memory.through));
                    ScrollArea::vertical().max_height(260.0).show(ui, |ui| {
                        ui.label(&memory.summary);
                    });
                }
            });
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, "workbench-v1", &self.saved);
    }
}

impl Drop for Workbench {
    fn drop(&mut self) {
        self.generation.take();
        self.engine.take();
    }
}

fn model_settings(ui: &mut Ui, cfg: &mut EngineConfig) {
    section(ui, "Вычисления");
    egui::ComboBox::from_label("Движок")
        .selected_text(cfg.backend.label())
        .show_ui(ui, |ui| {
            for backend in [
                Backend::Auto,
                Backend::Cuda,
                Backend::Cpu,
                Backend::Existing,
            ] {
                ui.selectable_value(&mut cfg.backend, backend, backend.label());
            }
        });
    if cfg.backend == Backend::Existing {
        path_setting(ui, "Исполняемый файл", &mut cfg.executable, false);
    } else {
        ui.collapsing("Инструменты сборки", |ui| {
            path_setting(ui, "Папка llama.cpp", &mut cfg.source, true);
            text(ui, "Команда или путь к CMake", &mut cfg.cmake);
        });
    }
    number(ui, "Контекст", &mut cfg.context, 1024..=131072);
    number(ui, "Батч", &mut cfg.batch, 32..=8192);
    cfg.ubatch = cfg.ubatch.min(cfg.batch);
    number(ui, "Микробатч", &mut cfg.ubatch, 16..=cfg.batch);
    number(ui, "Потоки CPU (0 = авто)", &mut cfg.threads, 0..=256);
    number(
        ui,
        "Потоки батча (0 = авто)",
        &mut cfg.batch_threads,
        0..=256,
    );
    if cfg.backend != Backend::Cpu {
        text(ui, "Слои GPU (auto, all, число)", &mut cfg.gpu_layers);
        number(ui, "Запас VRAM, МиБ", &mut cfg.fit_margin, 0..=24576);
    }
    choice(
        ui,
        "Flash Attention",
        &mut cfg.flash,
        &["auto", "on", "off"],
    );
    ui.checkbox(&mut cfg.warmup, "Прогрев модели");
    ui.checkbox(&mut cfg.context_shift, "Сдвиг окна контекста");

    section(ui, "Dory и KV-кэш");
    number(ui, "Круги (0 = из GGUF)", &mut cfg.dory_loops, 0..=4);
    choice(
        ui,
        "Режим кругов",
        &mut cfg.dory_kv,
        &["model", "last_loop", "per_loop"],
    );
    choice(ui, "Ключи K", &mut cfg.key_cache, &["f16", "q8_0", "q4_0"]);
    choice(
        ui,
        "Значения V",
        &mut cfg.value_cache,
        &["f16", "q8_0", "q4_0"],
    );
    ui.checkbox(&mut cfg.trace, "Показывать круги в реальном времени").on_hover_text("Диагностический режим синхронизирует вычисления между кругами и может замедлять генерацию.");

    section(ui, "Шаблон чата");
    path_setting(
        ui,
        "Пользовательский Jinja-шаблон",
        &mut cfg.template,
        false,
    );
    if !cfg.template.as_os_str().is_empty() && ui.small_button("Использовать шаблон GGUF").clicked()
    {
        cfg.template.clear();
    }
}

fn sampling_settings(ui: &mut Ui, cfg: &mut Sampling) {
    section(ui, "Сэмплирование");
    ui.add(egui::Slider::new(&mut cfg.temperature, 0.0..=2.0).text("Температура"));
    ui.add(egui::Slider::new(&mut cfg.top_p, 0.0..=1.0).text("Top P"));
    ui.add(egui::Slider::new(&mut cfg.min_p, 0.0..=1.0).text("Min P"));
    ui.add(egui::Slider::new(&mut cfg.repeat_penalty, 0.75..=2.0).text("Повторы"));
    number(ui, "Top K", &mut cfg.top_k, 0..=512);
    number(ui, "Длина ответа", &mut cfg.tokens, 1..=32768);
    number(ui, "Seed (-1 = случайный)", &mut cfg.seed, -1..=i32::MAX);

    section(ui, "Мышление");
    choice(
        ui,
        "Показывать/генерировать",
        &mut cfg.reasoning,
        &["auto", "on", "off"],
    );
    number(
        ui,
        "Бюджет (-1 = без лимита)",
        &mut cfg.reasoning_budget,
        -1..=32768,
    );

    section(ui, "Системная инструкция");
    ui.add(
        TextEdit::multiline(&mut cfg.system)
            .desired_rows(5)
            .desired_width(f32::INFINITY)
            .hint_text("Инструкции для модели"),
    );
    if ui.button("Сбросить настройки ответа").clicked() {
        *cfg = Sampling::default();
    }
}

fn path_setting(ui: &mut Ui, label: &str, path: &mut PathBuf, folder: bool) {
    ui.label(RichText::new(label).small().color(MUTE));
    ui.horizontal(|ui| {
        let mut display = path.to_string_lossy().into_owned();
        if ui
            .add(
                TextEdit::singleline(&mut display)
                    .desired_width((ui.available_width() - 42.0).max(70.0)),
            )
            .changed()
        {
            *path = PathBuf::from(display);
        }
        if ui.small_button("…").clicked() {
            let picker = rfd::FileDialog::new();
            let picked = if folder {
                picker.pick_folder()
            } else {
                picker.pick_file()
            };
            if let Some(picked) = picked {
                *path = picked;
            }
        }
    });
}

fn text(ui: &mut Ui, label: &str, value: &mut String) {
    ui.label(RichText::new(label).small().color(MUTE));
    ui.add(TextEdit::singleline(value).desired_width(f32::INFINITY));
}

fn choice(ui: &mut Ui, label: &str, value: &mut String, items: &[&str]) {
    egui::ComboBox::from_label(label)
        .selected_text(value.as_str())
        .show_ui(ui, |ui| {
            for item in items {
                ui.selectable_value(value, (*item).into(), *item);
            }
        });
}

fn number<T: egui::emath::Numeric>(
    ui: &mut Ui,
    label: &str,
    value: &mut T,
    range: std::ops::RangeInclusive<T>,
) {
    ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(value).range(range));
        ui.label(label);
    });
}

fn token_meter(ui: &mut Ui, used: Option<usize>, context: u32) {
    let label = used.map_or_else(
        || format!("Окно · {context} токенов"),
        |used| format!("{used} / {context} токенов"),
    );
    ui.add(
        egui::ProgressBar::new(used.unwrap_or(0) as f32 / context.max(1) as f32)
            .text(label)
            .fill(ACCENT),
    );
}

fn message_card(ui: &mut Ui, message: &Message) {
    ui.add_space(6.0);
    if message.role == "tool" {
        ui.collapsing("Вывод инструмента", |ui| {
            selectable_text(ui, &message.content, TextStyle::Monospace)
        });
        return;
    }
    let user = message.role == "user";
    Frame::new()
        .fill(if user { CARD } else { PANEL })
        .stroke(Stroke::new(1.0_f32, LINE))
        .corner_radius(9)
        .inner_margin(12)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(if user { "Ты" } else { "Dory" })
                        .strong()
                        .color(if user { TEXT } else { ACCENT }),
                );
                if !message.content.is_empty() {
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.small_button("Копировать").clicked() {
                            ui.ctx().copy_text(message.content.clone());
                        }
                    });
                }
            });
            if !message.reasoning_content.is_empty() {
                ui.collapsing("Ход рассуждения", |ui| {
                    selectable_text(ui, &message.reasoning_content, TextStyle::Body)
                });
            }
            if !message.content.is_empty() {
                selectable_text(ui, &message.content, TextStyle::Body);
            }
            for call in &message.tool_calls {
                ui.collapsing(&call.function.name, |ui| {
                    selectable_text(ui, &call.function.arguments, TextStyle::Monospace)
                });
            }
        });
}

fn selectable_text(ui: &mut Ui, mut text: &str, font: TextStyle) {
    ui.add(
        TextEdit::multiline(&mut text)
            .font(font)
            .desired_rows(1)
            .desired_width(f32::INFINITY)
            .frame(false),
    );
}

fn scan_models(dir: &Path) -> Vec<PathBuf> {
    if dir.as_os_str().is_empty() {
        return Vec::new();
    }
    let mut models = Vec::new();
    let mut dirs = vec![(dir.to_path_buf(), 0)];
    while let Some((dir, depth)) = dirs.pop() {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten().take(1000) {
                let path = entry.path();
                if is_model(&path) && !name(&path).starts_with("mmproj") {
                    if let Some((_, suffix)) = name(&path).rsplit_once("-of-")
                        && suffix.len() == 10
                        && !name(&path).contains("-00001-of-")
                    {
                        continue;
                    }
                    models.push(path);
                } else if depth < 2
                    && dirs.len() < 256
                    && entry.file_type().is_ok_and(|kind| kind.is_dir())
                {
                    dirs.push((path, depth + 1));
                }
            }
        }
        if models.len() >= 300 {
            break;
        }
    }
    models.sort();
    models.truncate(300);
    models
}

fn name(path: &Path) -> &str {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("GGUF не выбран")
}

fn heading(ui: &mut Ui, label: &str) {
    ui.label(RichText::new(label).size(16.0).strong());
}

fn section(ui: &mut Ui, label: &str) {
    ui.add_space(12.0);
    ui.label(RichText::new(label).strong().color(ACCENT));
    ui.add_space(5.0);
}

fn card<R>(ui: &mut Ui, render: impl FnOnce(&mut Ui) -> R) -> R {
    Frame::new()
        .fill(CARD)
        .stroke(Stroke::new(1.0_f32, LINE))
        .corner_radius(9)
        .inner_margin(11)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            render(ui)
        })
        .inner
}

fn primary(label: &str) -> egui::Button<'_> {
    egui::Button::new(RichText::new(label).color(INK).strong())
        .fill(ACCENT)
        .min_size(Vec2::new(96.0, 29.0))
}

fn style(ctx: &Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(8.0, 7.0);
    style.spacing.button_padding = Vec2::new(8.0, 5.0);
    style.spacing.indent = 15.0;
    style
        .text_styles
        .insert(TextStyle::Body, egui::FontId::proportional(15.0));
    style
        .text_styles
        .insert(TextStyle::Button, egui::FontId::proportional(14.0));
    let mut visual = egui::Visuals::dark();
    visual.override_text_color = Some(TEXT);
    visual.panel_fill = PANEL;
    visual.window_fill = PANEL;
    visual.faint_bg_color = CARD;
    visual.extreme_bg_color = BG;
    visual.selection.bg_fill = Color32::from_rgb(22, 79, 73);
    visual.selection.stroke = Stroke::new(1.0_f32, ACCENT);
    visual.widgets.inactive.bg_fill = CARD;
    visual.widgets.inactive.weak_bg_fill = CARD;
    visual.widgets.hovered.weak_bg_fill = Color32::from_rgb(39, 67, 77);
    visual.widgets.active.weak_bg_fill = Color32::from_rgb(30, 87, 79);
    visual.widgets.hovered.bg_fill = Color32::from_rgb(39, 67, 77);
    visual.widgets.active.bg_fill = Color32::from_rgb(30, 87, 79);
    visual.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, LINE);
    style.visuals = visual;
    ctx.set_style(style);
}
