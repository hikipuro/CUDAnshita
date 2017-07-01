using System;

namespace CUDAnshita {
	public class Context : IDisposable {
		IntPtr context = IntPtr.Zero;

		public uint ApiVersion {
			get { return NvCuda.CtxGetApiVersion(context); }
		}

		public CUfunc_cache CacheConfig {
			get { return NvCuda.CtxGetCacheConfig(); }
			set { NvCuda.CtxSetCacheConfig(value); }
		}

		public int Device {
			get { return NvCuda.CtxGetDevice(); }
		}

		public uint Flags {
			get { return NvCuda.CtxGetFlags(); }
		}

		internal Context(int deviceHandle) {
			context = NvCuda.CtxCreate(0, deviceHandle);
		}

		public void Dispose() {
			if (context != IntPtr.Zero) {
				NvCuda.CtxDestroy(context);
				context = IntPtr.Zero;
			}
		}

		public void Synchronize() {
			NvCuda.CtxSynchronize();
		}

		public void Push() {
			NvCuda.CtxPushCurrent(context);
		}

		public IntPtr Pop() {
			return NvCuda.CtxPopCurrent();
		}

		public void DisablePeerAccess(IntPtr peerContext) {
			NvCuda.CtxDisablePeerAccess(peerContext);
		}

		public void EnablePeerAccess(IntPtr peerContext, uint flags) {
			NvCuda.CtxEnablePeerAccess(peerContext, flags);
		}

		public long GetLimit(CUlimit limit) {
			return NvCuda.CtxGetLimit(limit);
		}

		public void SetLimit(CUlimit limit, long value) {
			NvCuda.CtxSetLimit(limit, value);
		}

		public CUsharedconfig GetSharedMemConfig(CUlimit limit) {
			return NvCuda.CtxGetSharedMemConfig(limit);
		}

		public void SetSharedMemConfig(CUsharedconfig config) {
			NvCuda.CtxSetSharedMemConfig(config);
		}
	}
}
