using System.Runtime.InteropServices;

namespace CUDAnshita {
	[StructLayout(LayoutKind.Sequential)]
	public struct __half {
		public ushort x;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct __half2 {
		public uint x;
	}
}
