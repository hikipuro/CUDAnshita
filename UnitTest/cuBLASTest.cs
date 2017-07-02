using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAnshita;

namespace UnitTest {
	[TestClass]
	public class cuBLASTest {
		[ClassInitialize]
		public static void ClassInit(TestContext context) {
			Runtime.DeviceReset();
		}

		[TestMethod]
		public void GetVersion_v2() {
			IntPtr handle = cuBLAS.Create_v2();
			int version = cuBLAS.GetVersion_v2(handle);
			Assert.AreNotEqual(0, version);
			cuBLAS.Destroy_v2(handle);
		}

		[TestMethod]
		public void SetVector() {
			int count = 10;
			float[] data = new float[count];
			for (int i = 0; i < count; i++) {
				data[i] = 0.01f * i;
			}
			IntPtr dm = Runtime.Malloc(sizeof(float) * count);

			cuBLAS.SetVector<float>(data, dm);
			float[] result = cuBLAS.GetVector<float>(count, dm);

			Assert.IsTrue(IsSameArray(data, result));
		}


		//[TestMethod]
		//public void Nrm2Ex() {
		//	IntPtr handle = cuBLAS.Create_v2();
		//	cuBLAS.Destroy_v2(handle);
		//}

		[TestMethod]
		public void Snrm2_v2() {
			IntPtr handle = cuBLAS.Create_v2();

			int count = 10;
			float[] data = new float[count];
			for (int i = 0; i < count; i++) {
				data[i] = 0.01f * i;
			}
			IntPtr dm = Runtime.Malloc(sizeof(float) * count);

			// host
			cuBLAS.SetVector(data, dm);
			float resultHost = cuBLAS.Snrm2_v2(handle, count, dm, 1);
			//Console.WriteLine("Test 1: {0}", resultHost);

			// device
			IntPtr resultDevice = Runtime.Malloc(sizeof(float));
			cuBLAS.Snrm2_v2(handle, count, dm, 1, resultDevice);
			float[] r = Runtime.MemcpyD2H<float>(resultDevice, 1);
			//Console.WriteLine("Test 2: {0}", r[0]);
			Runtime.Free(resultDevice);

			Assert.AreEqual(resultHost, r[0]);

			Runtime.Free(dm);
			cuBLAS.Destroy_v2(handle);
		}

		private bool IsSameArray<T>(T[] array1, T[] array2) {
			if (array1 == null || array2 == null) {
				return false;
			}
			if (array1.Length != array2.Length) {
				return false;
			}

			int length = array1.Length;
			for (int i = 0; i < length; i++) {
				if (array1[i].Equals(array2[i]) == false) {
					return false;
				}
			}
			return true;
		}
	}
}
