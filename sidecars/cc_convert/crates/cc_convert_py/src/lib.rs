//! Python bindings: JSON in, JSON out. The Python wrapper marshals
//! dict ↔ JSON so the binding surface stays tiny.

use cc_convert_core::{
    anthropic_request_to_openai, openai_response_to_anthropic, stream::StreamTranslator,
    tool_names::ToolNameMap, ConvertOptions,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

fn pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(format!("{}", e))
}

fn resolve_options(mode: Option<&str>, target_model: Option<String>) -> PyResult<ConvertOptions> {
    let mut opts = match mode.unwrap_or("pragmatic") {
        "pragmatic" => ConvertOptions::pragmatic(),
        "litellm_compat" | "litellm-compat" | "litellm" => ConvertOptions::litellm_compat(),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown mode {:?}; expected 'pragmatic' or 'litellm_compat'",
                other
            )))
        }
    };
    opts.target_model = target_model;
    Ok(opts)
}

#[pyfunction]
#[pyo3(signature = (anthropic_request_json, target_model=None, mode=None))]
fn translate_request(
    anthropic_request_json: &str,
    target_model: Option<String>,
    mode: Option<&str>,
) -> PyResult<(String, String)> {
    let req: cc_convert_core::anthropic::AnthropicRequest =
        serde_json::from_str(anthropic_request_json).map_err(pyerr)?;
    let opts = resolve_options(mode, target_model)?;
    let (openai_req, tool_map) = anthropic_request_to_openai(&req, &opts).map_err(pyerr)?;
    let openai_str = serde_json::to_string(&openai_req).map_err(pyerr)?;
    let map_str = serde_json::to_string(&tool_map).map_err(pyerr)?;
    Ok((openai_str, map_str))
}

#[pyfunction]
fn translate_response(
    openai_response_json: &str,
    original_model: &str,
    tool_map_json: &str,
) -> PyResult<String> {
    let resp: cc_convert_core::openai::OpenAIResponse =
        serde_json::from_str(openai_response_json).map_err(pyerr)?;
    let tool_map: ToolNameMap = serde_json::from_str(tool_map_json).map_err(pyerr)?;
    let anthropic_resp =
        openai_response_to_anthropic(&resp, original_model, &tool_map).map_err(pyerr)?;
    Ok(serde_json::to_string(&anthropic_resp).map_err(pyerr)?)
}

#[pyclass]
struct PyStreamTranslator {
    inner: StreamTranslator,
}

#[pymethods]
impl PyStreamTranslator {
    #[new]
    fn new(original_model: String, tool_map_json: &str) -> PyResult<Self> {
        let tool_map: ToolNameMap = serde_json::from_str(tool_map_json).map_err(pyerr)?;
        Ok(Self {
            inner: StreamTranslator::new(original_model, tool_map),
        })
    }

    fn push(&mut self, openai_chunk_json: &str) -> PyResult<Vec<String>> {
        let chunk: cc_convert_core::openai::OpenAIStreamChunk =
            serde_json::from_str(openai_chunk_json).map_err(pyerr)?;
        let events = self.inner.push_openai_chunk(&chunk);
        events
            .iter()
            .map(|e| serde_json::to_string(e).map_err(pyerr))
            .collect()
    }

    fn finish(&mut self) -> PyResult<Vec<String>> {
        let events = self.inner.finish();
        events
            .iter()
            .map(|e| serde_json::to_string(e).map_err(pyerr))
            .collect()
    }
}

#[pymodule]
#[pyo3(name = "_native")]
fn cc_convert_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(translate_request, m)?)?;
    m.add_function(wrap_pyfunction!(translate_response, m)?)?;
    m.add_class::<PyStreamTranslator>()?;
    m.add("__version__", PyString::new_bound(_py, env!("CARGO_PKG_VERSION")))?;
    Ok(())
}
