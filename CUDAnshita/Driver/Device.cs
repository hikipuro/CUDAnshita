using System;

namespace CUDAnshita {
	public class Device {
		static bool initialized = false;

		int device = 0;

		public static int Count {
			get { return Driver.DeviceGetCount(); }
		}

		public long TotalMem {
			get { return Driver.DeviceTotalMem(device); }
		}

		public string Name {
			get { return Driver.DeviceGetName(device); }
		}

		public string PCIBusId {
			get { return Driver.DeviceGetPCIBusId(device); }
		}

		public Device(int deviceNumber) {
			if (initialized == false) {
				initialized = true;
				Driver.Init(0);
			}
			device = Driver.DeviceGet(deviceNumber);
		}

		public Context CreateContext() {
			return new Context(device);
		}

		public Context GetCurrentContext() {
			return new Context(Driver.CtxGetCurrent());
		}

		public int GetAttribute(CUdevice_attribute attrib) {
			return Driver.DeviceGetAttribute(attrib, device);
		}

		//public CUdevprop GetProperties() {
		//	return Driver.DeviceGetProperties(device);
		//}
	}
}
