use std::sync::{Arc, RwLock};

use flume::Receiver;
use uuid::Uuid;
use wgpu::BufferAsyncError;

pub struct Context {
    pub(crate) id: Uuid,
    pub(crate) executor: Arc<RwLock<Executor>>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            executor: Arc::new(RwLock::new(Executor::new())),
        }
    }
}

pub(crate) struct Buffers {
    pub(crate) receiver: Option<Receiver<Result<(), BufferAsyncError>>>,
    pub(crate) storage: wgpu::Buffer,
    pub(crate) staging: wgpu::Buffer,
}

pub struct Executor {
    pub(crate) synced: bool,
    pub(crate) device: wgpu::Device,
    pub(crate) encoder: wgpu::CommandEncoder,
    pub(crate) queue: wgpu::Queue,
}

impl Executor {
    pub fn new() -> Self {
        let (device, queue) = pollster::block_on(Self::create_device());
        let encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        Self {
            synced: false,
            device,
            queue,
            encoder,
        }
    }

    async fn create_device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();

        let mut limits = wgpu::Limits::downlevel_defaults();

        let features = adapter.features();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features,
                    limits,
                },
                None,
            )
            .await
            .unwrap();
        (device, queue)
    }

    pub(crate) fn sync(&mut self) {
        if self.synced {
            return;
        }

        // Actually poll GPU here
        let current_encoder = std::mem::replace(
            &mut self.encoder,
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }),
        );
        self.queue.submit(Some(current_encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // update state
        self.synced = true;
    }
}
