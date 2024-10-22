use rspack_collections::Identifier;
use rspack_core::{impl_runtime_module, Compilation, RuntimeGlobals, RuntimeModule};

#[impl_runtime_module]
#[derive(Debug)]
pub struct NonceRuntimeModule {
  id: Identifier,
}

impl Default for NonceRuntimeModule {
  fn default() -> Self {
    Self::with_default(Identifier::from("webpack/runtime/nonce"))
  }
}

impl RuntimeModule for NonceRuntimeModule {
  fn name(&self) -> Identifier {
    self.id
  }

  fn generate(&self, _: &Compilation) -> rspack_error::Result<String> {
    Ok(format!("{} = undefined;", RuntimeGlobals::SCRIPT_NONCE).to_string())
  }
}
