use rspack_collections::Identifier;
use rspack_core::{
  get_chunk_from_ukey, impl_runtime_module, ChunkUkey, Compilation, RuntimeGlobals, RuntimeModule,
};

#[impl_runtime_module]
#[derive(Debug)]
pub struct BaseUriRuntimeModule {
  id: Identifier,
  chunk: Option<ChunkUkey>,
}

impl Default for BaseUriRuntimeModule {
  fn default() -> Self {
    Self::with_default(Identifier::from("webpack/runtime/base_uri"), None)
  }
}

impl RuntimeModule for BaseUriRuntimeModule {
  fn name(&self) -> Identifier {
    self.id
  }

  fn attach(&mut self, chunk: ChunkUkey) {
    self.chunk = Some(chunk);
  }

  fn generate(&self, compilation: &Compilation) -> rspack_error::Result<String> {
    let base_uri = self
      .chunk
      .and_then(|ukey| get_chunk_from_ukey(&ukey, &compilation.chunk_by_ukey))
      .and_then(|chunk| chunk.get_entry_options(&compilation.chunk_group_by_ukey))
      .and_then(|options| options.base_uri.as_ref())
      .and_then(|base_uri| serde_json::to_string(base_uri).ok())
      .unwrap_or_else(|| "undefined".to_string());
    Ok(format!("{} = {};\n", RuntimeGlobals::BASE_URI, base_uri))
  }
}
