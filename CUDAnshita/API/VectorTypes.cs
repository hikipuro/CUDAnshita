using System.Runtime.InteropServices;

namespace CUDAnshita {
	[StructLayout(LayoutKind.Sequential)]
	public struct char1 {
		public sbyte x;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct char2 {
		public sbyte x;
		public sbyte y;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct char3 {
		public sbyte x;
		public sbyte y;
		public sbyte z;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct char4 {
		public sbyte x;
		public sbyte y;
		public sbyte z;
		public sbyte w;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct uchar1 {
		public byte x;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct uchar2 {
		public byte x;
		public byte y;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct uchar3 {
		public byte x;
		public byte y;
		public byte z;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct uchar4 {
		public byte x;
		public byte y;
		public byte z;
		public byte w;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct short1 {
		public short x;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct short2 {
		public short x;
		public short y;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct short3 {
		public short x;
		public short y;
		public short z;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct short4 {
		public short x;
		public short y;
		public short z;
		public short w;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct ushort1 {
		public ushort x;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct ushort2 {
		public ushort x;
		public ushort y;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct ushort3 {
		public ushort x;
		public ushort y;
		public ushort z;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct ushort4 {
		public ushort x;
		public ushort y;
		public ushort z;
		public ushort w;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct int1 {
		public int x;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct int2 {
		public int x;
		public int y;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct int3 {
		public int x;
		public int y;
		public int z;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct int4 {
		public int x;
		public int y;
		public int z;
		public int w;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct uint1 {
		public uint x;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct uint2 {
		public uint x;
		public uint y;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct uint3 {
		public uint x;
		public uint y;
		public uint z;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct uint4 {
		public uint x;
		public uint y;
		public uint z;
		public uint w;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct long1 {
		public int x;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct long2 {
		public int x;
		public int y;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct long3 {
		public int x;
		public int y;
		public int z;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct long4 {
		public int x;
		public int y;
		public int z;
		public int w;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct ulong1 {
		public uint x;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct ulong2 {
		public uint x;
		public uint y;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct ulong3 {
		public uint x;
		public uint y;
		public uint z;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct ulong4 {
		public uint x;
		public uint y;
		public uint z;
		public uint w;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct longlong1 {
		public long x;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct longlong2 {
		public long x;
		public long y;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct longlong3 {
		public long x;
		public long y;
		public long z;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct longlong4 {
		public long x;
		public long y;
		public long z;
		public long w;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct ulonglong1 {
		public ulong x;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct ulonglong2 {
		public ulong x;
		public ulong y;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct ulonglong3 {
		public ulong x;
		public ulong y;
		public ulong z;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct ulonglong4 {
		public ulong x;
		public ulong y;
		public ulong z;
		public ulong w;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct float1 {
		public float x;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct float2 {
		public float x;
		public float y;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct float3 {
		public float x;
		public float y;
		public float z;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct float4 {
		public float x;
		public float y;
		public float z;
		public float w;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct double1 {
		public double x;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct double2 {
		public double x;
		public double y;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct double3 {
		public double x;
		public double y;
		public double z;
	}
	[StructLayout(LayoutKind.Sequential)]
	public struct double4 {
		public double x;
		public double y;
		public double z;
		public double w;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct dim3 {
		public uint x, y, z;

		public dim3(uint x = 1, uint y = 1, uint z = 1) {
			this.x = x;
			this.y = y;
			this.z = z;
		}
	}
}
