using System;
using System.Text;

namespace CUDAnshita {
	public class CudaMatrixFloat {
		const int ItemSize = sizeof(float);

		protected int _Cols;
		protected int _Rows;
		protected IntPtr _DevicePointer;

		const string MatrixProgram = @"
extern ""C"" {
	__global__ void matrixAdd(float *a, float *b, float c, int width, int height) {
		int x = blockDim.x * blockIdx.x + threadIdx.x;
		int y = blockDim.y * blockIdx.y + threadIdx.y;
		if (x >= width || y >= height) {
			return;
		}
		int index = y * width + x;
		b[index] = a[index] + c;
	}
}
";
		protected static Module _Module;

		string Compile(string name, string src) {
			RuntimeCompiler compiler = new RuntimeCompiler();
			compiler.AddOptions(
				RuntimeCompiler.OPTION_TARGET_20
				//RuntimeCompiler.OPTION_FMAD_FALSE,
				//RuntimeCompiler.OPTION_LINE_INFO,
				//RuntimeCompiler.OPTION_DEVICE_AS_DEFAULT_EXECUTION_SPACE
			);

			string ptx = compiler.Compile(name, src);
			if (ptx == null) {
				Console.WriteLine("Compile Error:");
				Console.WriteLine();
				Console.WriteLine(compiler.Log);
				return null;
			}
			return ptx;
		}

		void CallMethod(Module module, string name, int width, int height, params object[] args) {
			int threadX = 128;
			int threadY = 2;
			int blockX = divRoundUp(width, threadX);
			int blockY = divRoundUp(height, threadY);

			module.SetBlockCount(blockX, blockY, 1);
			module.SetThreadCount(threadX, threadY, 1);
			module.Excecute(name, args);
		}

		int divRoundUp(int value, int radix) {
			return (value + radix - 1) / radix;
		}

		Module CreateModule(string ptx) {
			if (ptx == null) {
				return null;
			}
			Module module = new Module();
			module.LoadData(ptx);
			return module;
		}


		public int Cols {
			get { return _Cols; }
		}

		public int Rows {
			get { return _Rows; }
		}

		public int Count {
			get { return _Cols * _Rows; }
		}

		public CudaMatrixFloat(int rows, int cols) {
			if (rows < 1) {
				rows = 1;
			}
			if (cols < 1) {
				cols = 1;
			}
			_Rows = rows;
			_Cols = cols;
			_DevicePointer = MallocDeviceMemory(Count);

			if (_Module == null) {
				string ptx = Compile("CudaMatrixFloat.cu", MatrixProgram);
				_Module = CreateModule(ptx);
			}
		}

		public CudaMatrixFloat(int rows, int cols, float[] data) : this(rows, cols) {
			if (data == null) {
				throw new ArgumentNullException();
			}
			if (rows * cols < data.Length) {
				throw new ArgumentException();
			}
			SetDeviceMemory(_DevicePointer, data);
		}

		public float[] GetData() {
			return GetDeviceMemory(_DevicePointer, Count);
		}

		public void Clear() {
			Runtime.Memset(_DevicePointer, 0, ItemSize * Count);
		}

		public void CopyTo(CudaMatrixFloat result) {
			if (result == null) {
				return;
			}

			IntPtr handle = cuBLAS.Create_v2();
			cuBLAS.Scopy_v2(
				handle, Count,
				_DevicePointer, 1
				, result._DevicePointer, 1
			);
			cuBLAS.Destroy_v2(handle);
		}

		public void Add(float value, CudaMatrixFloat result) {
			if (result == null) {
				return;
			}
			if (_Module == null) {
				throw new CudaException("module is not initialized.");
			}

			CallMethod(
				_Module, "matrixAdd",
				_Cols, _Rows,
				_DevicePointer,
				result._DevicePointer,
				value,
				_Cols, _Rows
			);
		}

		public void Add(CudaMatrixFloat matrix, CudaMatrixFloat result) {
			if (matrix == null || result == null) {
				return;
			}

			IntPtr handle = cuBLAS.Create_v2();
			cuBLAS.Sgeam(
				handle,
				cublasOperation.CUBLAS_OP_N,
				cublasOperation.CUBLAS_OP_N,
				_Rows, _Cols,
				1f,
				_DevicePointer, _Rows,
				1f,
				matrix._DevicePointer, _Rows,
				result._DevicePointer, _Rows
			);
			cuBLAS.Destroy_v2(handle);
		}

		public void Sub(CudaMatrixFloat matrix, CudaMatrixFloat result) {
			if (matrix == null || result == null) {
				return;
			}

			IntPtr handle = cuBLAS.Create_v2();
			cuBLAS.Sgeam(
				handle,
				cublasOperation.CUBLAS_OP_N,
				cublasOperation.CUBLAS_OP_N,
				_Rows, _Cols,
				1f,
				_DevicePointer, _Rows,
				-1f,
				matrix._DevicePointer, _Rows,
				result._DevicePointer, _Rows
			);
			cuBLAS.Destroy_v2(handle);
		}

		public void Mul(float value, CudaMatrixFloat result) {
			if (result == null) {
				return;
			}

			IntPtr handle = cuBLAS.Create_v2();
			cuBLAS.Sscal_v2(
				handle,
				_Rows * _Cols,
				value,
				_DevicePointer,
				1
			);
			cuBLAS.Destroy_v2(handle);
		}

		public void Mul(CudaMatrixFloat matrix, CudaMatrixFloat result) {
			if (matrix == null || result == null) {
				return;
			}

			IntPtr handle = cuBLAS.Create_v2();
			cuBLAS.Sdgmm(
				handle,
				cublasSideMode.CUBLAS_SIDE_LEFT,
				_Rows, _Cols,
				_DevicePointer, _Rows,
				matrix._DevicePointer, _Rows,
				result._DevicePointer, _Rows
			);
			cuBLAS.Destroy_v2(handle);
		}

		public void Dot(CudaMatrixFloat matrix, CudaMatrixFloat result) {
			if (matrix == null || result == null) {
				return;
			}

			IntPtr handle = cuBLAS.Create_v2();
			cuBLAS.Sgemm_v2(
				handle,
				cublasOperation.CUBLAS_OP_N,
				cublasOperation.CUBLAS_OP_N,
				_Rows, matrix._Cols, _Cols,
				1f,
				_DevicePointer, _Rows,
				matrix._DevicePointer, matrix._Rows,
				1f,
				result._DevicePointer, _Rows
			);
			cuBLAS.Destroy_v2(handle);
		}

		public float Sum() {
			IntPtr handle = cuBLAS.Create_v2();
			float result = cuBLAS.Sasum_v2(
				handle,
				Count,
				_DevicePointer, 1
			);
			cuBLAS.Destroy_v2(handle);
			return result;
		}

		public int MaxIndex() {
			IntPtr handle = cuBLAS.Create_v2();
			int index = cuBLAS.Isamax_v2(
				handle,
				Count,
				_DevicePointer, 1
			);
			cuBLAS.Destroy_v2(handle);
			return index - 1;
		}

		public int MinIndex() {
			IntPtr handle = cuBLAS.Create_v2();
			int index = cuBLAS.Isamin_v2(
				handle,
				Count,
				_DevicePointer, 1
			);
			cuBLAS.Destroy_v2(handle);
			return index - 1;
		}


		public static CudaMatrixFloat FromByteArray(byte[] bytes) {
			if (bytes == null || bytes.Length < 12) {
				return null;
			}

			int intSize = sizeof(int);
			int intSize2 = intSize * 2;
			int[] matrixSize = new int[2];
			float[] data = new float[(bytes.Length - intSize2) / ItemSize];
			Buffer.BlockCopy(bytes, 0, matrixSize, 0, intSize2);
			Buffer.BlockCopy(bytes, intSize2, data, 0, bytes.Length - intSize2);

			int rows = matrixSize[0];
			int cols = matrixSize[1];
			CudaMatrixFloat matrix = new CudaMatrixFloat(rows, cols);
			matrix.SetDeviceMemory(matrix._DevicePointer, data);
			return matrix;
		}

		public byte[] ToByteArray() {
			float[] data = GetData();
			int intSize = sizeof(int);
			int intSize2 = intSize * 2;
			int size = intSize2 + data.Length * ItemSize;
			byte[] bytes = new byte[size];
			Buffer.BlockCopy(new int[] { _Rows, _Cols }, 0, bytes, 0, intSize2);
			Buffer.BlockCopy(data, 0, bytes, intSize2, bytes.Length - intSize2);
			return bytes;
		}

		public override string ToString() {
			float[] data = GetData();
			StringBuilder text = new StringBuilder();
			text.Append("[");

			int width = _Cols;
			int height = _Rows;
			for (int y = 0; y < height; y++) {
				if (height > 1) {
					text.AppendLine();
					text.Append("\t[");
				}
				for (int x = 0; x < width; x++) {
					text.Append(data[x + y * _Cols].ToString());
					if (x + 1 != _Cols) {
						text.Append(", ");
					}
				}
				if (height > 1) {
					text.Append("]");
				}
			}
			if (height > 1) {
				text.AppendLine();
			}
			text.Append("]");
			return text.ToString();
		}

		protected IntPtr MallocDeviceMemory(int count) {
			IntPtr devicePointer = Runtime.Malloc(ItemSize * count);
			Runtime.Memset(devicePointer, 0, ItemSize * count);
			return devicePointer;
		}

		protected void SetDeviceMemory(IntPtr devicePointer, float[] data) {
			Runtime.MemcpyH2D(devicePointer, data);
		}

		protected float[] GetDeviceMemory(IntPtr devicePointer, int count) {
			return Runtime.MemcpyD2H(devicePointer, count);
		}
	}
}
