use std::{ptr::NonNull, sync::Arc};

use napi::Either;
use napi_derive::napi;
use rspack_core::{Compilation, ExportsInfo, ModuleGraph, RuntimeSpec};

#[napi]
pub struct JsExportsInfo {
  exports_info: ExportsInfo,
  compilation: NonNull<Compilation>,
}

impl JsExportsInfo {
  pub fn new(exports_info: ExportsInfo, compilation: &Compilation) -> Self {
    #[allow(clippy::unwrap_used)]
    Self {
      exports_info,
      compilation: NonNull::new(compilation as *const Compilation as *mut Compilation).unwrap(),
    }
  }

  fn as_mut(&mut self) -> napi::Result<ModuleGraph<'static>> {
    let compilation = unsafe { self.compilation.as_mut() };
    let module_graph = compilation.get_module_graph_mut();
    Ok(module_graph)
  }
}

#[napi]
impl JsExportsInfo {
  #[napi]
  pub fn set_used_in_unknown_way(
    &mut self,
    js_runtime: Option<Either<String, Vec<String>>>,
  ) -> napi::Result<bool> {
    let mut module_graph = self.as_mut()?;
    let runtime: Option<RuntimeSpec> = js_runtime.map(|js_rt| match js_rt {
      Either::A(str) => vec![str].into_iter().map(Arc::from).collect(),
      Either::B(vec) => vec.into_iter().map(Arc::from).collect(),
    });
    Ok(
      self
        .exports_info
        .set_used_in_unknown_way(&mut module_graph, runtime.as_ref()),
    )
  }
}