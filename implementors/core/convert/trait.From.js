(function() {var implementors = {
"wasmedge_sdk":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;CallingFrame&gt; for <a class=\"struct\" href=\"wasmedge_sdk/struct.Caller.html\" title=\"struct wasmedge_sdk::Caller\">Caller</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"wasmedge_sdk/types/enum.Val.html\" title=\"enum wasmedge_sdk::types::Val\">Val</a>&gt; for WasmValue"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;WasmValue&gt; for <a class=\"enum\" href=\"wasmedge_sdk/types/enum.Val.html\" title=\"enum wasmedge_sdk::types::Val\">Val</a>"]],
"wasmedge_sys":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;FuncType&gt; for <a class=\"struct\" href=\"wasmedge_sys/struct.FuncType.html\" title=\"struct wasmedge_sys::FuncType\">FuncType</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"wasmedge_sys/struct.FuncType.html\" title=\"struct wasmedge_sys::FuncType\">FuncType</a>&gt; for FuncType"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;GlobalType&gt; for <a class=\"struct\" href=\"wasmedge_sys/struct.GlobalType.html\" title=\"struct wasmedge_sys::GlobalType\">GlobalType</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"wasmedge_sys/struct.GlobalType.html\" title=\"struct wasmedge_sys::GlobalType\">GlobalType</a>&gt; for GlobalType"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;MemoryType&gt; for <a class=\"struct\" href=\"wasmedge_sys/struct.MemType.html\" title=\"struct wasmedge_sys::MemType\">MemType</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"wasmedge_sys/struct.MemType.html\" title=\"struct wasmedge_sys::MemType\">MemType</a>&gt; for MemoryType"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;TableType&gt; for <a class=\"struct\" href=\"wasmedge_sys/struct.TableType.html\" title=\"struct wasmedge_sys::TableType\">TableType</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"wasmedge_sys/struct.TableType.html\" title=\"struct wasmedge_sys::TableType\">TableType</a>&gt; for TableType"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.u32.html\">u32</a>&gt; for <a class=\"enum\" href=\"wasmedge_sys/plugin/enum.ProgramOptionType.html\" title=\"enum wasmedge_sys::plugin::ProgramOptionType\">ProgramOptionType</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"wasmedge_sys/plugin/enum.ProgramOptionType.html\" title=\"enum wasmedge_sys::plugin::ProgramOptionType\">ProgramOptionType</a>&gt; for <a class=\"type\" href=\"wasmedge_sys/ffi/type.WasmEdge_ProgramOptionType.html\" title=\"type wasmedge_sys::ffi::WasmEdge_ProgramOptionType\">WasmEdge_ProgramOptionType</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"wasmedge_sys/plugin/struct.PluginVersion.html\" title=\"struct wasmedge_sys::plugin::PluginVersion\">PluginVersion</a>&gt; for <a class=\"struct\" href=\"wasmedge_sys/ffi/struct.WasmEdge_PluginVersionData.html\" title=\"struct wasmedge_sys::ffi::WasmEdge_PluginVersionData\">WasmEdge_PluginVersionData</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"wasmedge_sys/ffi/struct.WasmEdge_String.html\" title=\"struct wasmedge_sys::ffi::WasmEdge_String\">WasmEdge_String</a>&gt; for <a class=\"struct\" href=\"https://doc.rust-lang.org/1.68.2/alloc/string/struct.String.html\" title=\"struct alloc::string::String\">String</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"wasmedge_sys/ffi/struct.WasmEdge_Value.html\" title=\"struct wasmedge_sys::ffi::WasmEdge_Value\">WasmEdge_Value</a>&gt; for <a class=\"struct\" href=\"wasmedge_sys/struct.WasmValue.html\" title=\"struct wasmedge_sys::WasmValue\">WasmValue</a>"]],
"wasmedge_types":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.68.2/alloc/ffi/c_str/struct.NulError.html\" title=\"struct alloc::ffi::c_str::NulError\">NulError</a>&gt; for <a class=\"enum\" href=\"wasmedge_types/error/enum.WasmEdgeError.html\" title=\"enum wasmedge_types::error::WasmEdgeError\">WasmEdgeError</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.68.2/core/ffi/c_str/struct.FromBytesWithNulError.html\" title=\"struct core::ffi::c_str::FromBytesWithNulError\">FromBytesWithNulError</a>&gt; for <a class=\"enum\" href=\"wasmedge_types/error/enum.WasmEdgeError.html\" title=\"enum wasmedge_types::error::WasmEdgeError\">WasmEdgeError</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.68.2/core/str/error/struct.Utf8Error.html\" title=\"struct core::str::error::Utf8Error\">Utf8Error</a>&gt; for <a class=\"enum\" href=\"wasmedge_types/error/enum.WasmEdgeError.html\" title=\"enum wasmedge_types::error::WasmEdgeError\">WasmEdgeError</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.68.2/alloc/string/struct.FromUtf8Error.html\" title=\"struct alloc::string::FromUtf8Error\">FromUtf8Error</a>&gt; for <a class=\"enum\" href=\"wasmedge_types/error/enum.WasmEdgeError.html\" title=\"enum wasmedge_types::error::WasmEdgeError\">WasmEdgeError</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.u32.html\">u32</a>&gt; for <a class=\"enum\" href=\"wasmedge_types/enum.RefType.html\" title=\"enum wasmedge_types::RefType\">RefType</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"wasmedge_types/enum.RefType.html\" title=\"enum wasmedge_types::RefType\">RefType</a>&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.u32.html\">u32</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.i32.html\">i32</a>&gt; for <a class=\"enum\" href=\"wasmedge_types/enum.RefType.html\" title=\"enum wasmedge_types::RefType\">RefType</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"wasmedge_types/enum.RefType.html\" title=\"enum wasmedge_types::RefType\">RefType</a>&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.i32.html\">i32</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.u32.html\">u32</a>&gt; for <a class=\"enum\" href=\"wasmedge_types/enum.ValType.html\" title=\"enum wasmedge_types::ValType\">ValType</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"wasmedge_types/enum.ValType.html\" title=\"enum wasmedge_types::ValType\">ValType</a>&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.u32.html\">u32</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.i32.html\">i32</a>&gt; for <a class=\"enum\" href=\"wasmedge_types/enum.ValType.html\" title=\"enum wasmedge_types::ValType\">ValType</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"wasmedge_types/enum.ValType.html\" title=\"enum wasmedge_types::ValType\">ValType</a>&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.i32.html\">i32</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.u32.html\">u32</a>&gt; for <a class=\"enum\" href=\"wasmedge_types/enum.Mutability.html\" title=\"enum wasmedge_types::Mutability\">Mutability</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"wasmedge_types/enum.Mutability.html\" title=\"enum wasmedge_types::Mutability\">Mutability</a>&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.u32.html\">u32</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.i32.html\">i32</a>&gt; for <a class=\"enum\" href=\"wasmedge_types/enum.Mutability.html\" title=\"enum wasmedge_types::Mutability\">Mutability</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"wasmedge_types/enum.Mutability.html\" title=\"enum wasmedge_types::Mutability\">Mutability</a>&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.i32.html\">i32</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.u32.html\">u32</a>&gt; for <a class=\"enum\" href=\"wasmedge_types/enum.CompilerOptimizationLevel.html\" title=\"enum wasmedge_types::CompilerOptimizationLevel\">CompilerOptimizationLevel</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"wasmedge_types/enum.CompilerOptimizationLevel.html\" title=\"enum wasmedge_types::CompilerOptimizationLevel\">CompilerOptimizationLevel</a>&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.u32.html\">u32</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.i32.html\">i32</a>&gt; for <a class=\"enum\" href=\"wasmedge_types/enum.CompilerOptimizationLevel.html\" title=\"enum wasmedge_types::CompilerOptimizationLevel\">CompilerOptimizationLevel</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"wasmedge_types/enum.CompilerOptimizationLevel.html\" title=\"enum wasmedge_types::CompilerOptimizationLevel\">CompilerOptimizationLevel</a>&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.i32.html\">i32</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.u32.html\">u32</a>&gt; for <a class=\"enum\" href=\"wasmedge_types/enum.CompilerOutputFormat.html\" title=\"enum wasmedge_types::CompilerOutputFormat\">CompilerOutputFormat</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"wasmedge_types/enum.CompilerOutputFormat.html\" title=\"enum wasmedge_types::CompilerOutputFormat\">CompilerOutputFormat</a>&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.u32.html\">u32</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.i32.html\">i32</a>&gt; for <a class=\"enum\" href=\"wasmedge_types/enum.CompilerOutputFormat.html\" title=\"enum wasmedge_types::CompilerOutputFormat\">CompilerOutputFormat</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"wasmedge_types/enum.CompilerOutputFormat.html\" title=\"enum wasmedge_types::CompilerOutputFormat\">CompilerOutputFormat</a>&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.i32.html\">i32</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.u32.html\">u32</a>&gt; for <a class=\"enum\" href=\"wasmedge_types/enum.HostRegistration.html\" title=\"enum wasmedge_types::HostRegistration\">HostRegistration</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"wasmedge_types/enum.HostRegistration.html\" title=\"enum wasmedge_types::HostRegistration\">HostRegistration</a>&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.u32.html\">u32</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.u32.html\">u32</a>&gt; for <a class=\"enum\" href=\"wasmedge_types/enum.ExternalInstanceType.html\" title=\"enum wasmedge_types::ExternalInstanceType\">ExternalInstanceType</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.68.2/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.68.2/std/primitive.i32.html\">i32</a>&gt; for <a class=\"enum\" href=\"wasmedge_types/enum.ExternalInstanceType.html\" title=\"enum wasmedge_types::ExternalInstanceType\">ExternalInstanceType</a>"]]
};if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()