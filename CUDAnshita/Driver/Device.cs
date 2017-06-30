using System;

namespace CUDAnshita {
	public class Device : IDisposable {
		static bool initialized = false;

		int device = 0;

		public static int Count {
			get { return NvCuda.DeviceGetCount(); }
		}

		public long TotalMem {
			get { return NvCuda.DeviceTotalMem(device); }
		}

		public string Name {
			get { return NvCuda.DeviceGetName(device); }
		}

		public string PCIBusId {
			get { return NvCuda.DeviceGetPCIBusId(device); }
		}

		public Device(int deviceNumber) {
			if (initialized == false) {
				initialized = true;
				NvCuda.Init(0);
			}
			device = NvCuda.DeviceGet(deviceNumber);
		}

		public void Dispose() {
			
		}

		public Context CreateContext() {
			return new Context(device);
		}

		public int GetAttribute(CUdevice_attribute attrib) {
			return NvCuda.DeviceGetAttribute(attrib, device);
		}

		public cudaDeviceProp GetProperties() {
			return CudaRT.GetDeviceProperties(device);
		}
	}
}
