// A simple wrapper around File::open adding details about the
// problematic file.
use std::path::Path;
use tch::Device;

pub(crate) fn file_open<P: AsRef<Path>>(path: P) -> anyhow::Result<std::fs::File> {
    std::fs::File::open(path.as_ref()).map_err(|e| {
        let context = format!("error opening {:?}", path.as_ref().to_string_lossy());
        anyhow::Error::new(e).context(context)
    })
}

pub struct DeviceSetup {
    accelerator_device: Device,
    cpu: Vec<String>,
}

impl DeviceSetup {
    pub fn new(cpu: Vec<String>) -> Self {
        let accelerator_device =
            if tch::utils::has_mps() { Device::Mps } else { Device::cuda_if_available() };
        Self { accelerator_device, cpu }
    }

    pub fn get(&self, name: &str) -> Device {
        if self.cpu.iter().any(|c| c == "all" || c == name) {
            Device::Cpu
        } else {
            self.accelerator_device
        }
    }
}
