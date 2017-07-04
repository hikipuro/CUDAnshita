using System;

namespace CUDAnshita {
	public class Context : IDisposable {
		IntPtr context = IntPtr.Zero;

		public uint ApiVersion {
			get { return Driver.CtxGetApiVersion(context); }
		}

		public CUfunc_cache CacheConfig {
			get { return Driver.CtxGetCacheConfig(); }
			set { Driver.CtxSetCacheConfig(value); }
		}

		public int Device {
			get { return Driver.CtxGetDevice(); }
		}

		public uint Flags {
			get { return Driver.CtxGetFlags(); }
		}

		internal Context(int deviceHandle) {
			context = Driver.CtxCreate(0, deviceHandle);
		}

		internal Context(IntPtr context) {
			this.context = context;
		}

		public void Dispose() {
			if (context != IntPtr.Zero) {
				Driver.CtxDestroy(context);
				context = IntPtr.Zero;
			}
		}

		public void Synchronize() {
			Driver.CtxSynchronize();
		}

		public void Push() {
			Driver.CtxPushCurrent(context);
		}

		public IntPtr Pop() {
			return Driver.CtxPopCurrent();
		}

		public void DisablePeerAccess(IntPtr peerContext) {
			Driver.CtxDisablePeerAccess(peerContext);
		}

		public void EnablePeerAccess(IntPtr peerContext, uint flags) {
			Driver.CtxEnablePeerAccess(peerContext, flags);
		}

		public long GetLimit(CUlimit limit) {
			return Driver.CtxGetLimit(limit);
		}

		public void SetLimit(CUlimit limit, long value) {
			Driver.CtxSetLimit(limit, value);
		}

		public CUsharedconfig GetSharedMemConfig(CUlimit limit) {
			return Driver.CtxGetSharedMemConfig(limit);
		}

		public void SetSharedMemConfig(CUsharedconfig config) {
			Driver.CtxSetSharedMemConfig(config);
		}
	}
}
