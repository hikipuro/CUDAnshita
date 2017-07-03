using System;

namespace CUDAnshita {
	public class CudaMatrixDouble : CudaMatrixBase {
		const int ItemSize = sizeof(double);
		double[] _Host;

		public double[] Data {
			get { return _Host; }
		}

		public double this[int i] {
			get { return _Host[i]; }
			set {
				_Dirty = true;
				_Host[i] = value;
			}
		}

		public double this[int x, int y] {
			get { return _Host[y * _Cols + x]; }
			set {
				_Dirty = true;
				_Host[y * _Cols + x] = value;
			}
		}

		public CudaMatrixDouble(int rows, int cols) : this(rows, cols, true) {
		}

		public CudaMatrixDouble(int rows, int cols, bool hostAlloc) {
			_Rows = rows;
			_Cols = cols;

			if (hostAlloc) {
				_Host = new double[Count];
			}
			_Device = Runtime.Malloc(ItemSize * Count);
			Runtime.Memset(_Device, 0, ItemSize * Count);
		}

		public static CudaMatrixDouble FromByteArray(byte[] bytes) {
			if (bytes == null || bytes.Length < 12) {
				return null;
			}

			int intSize = sizeof(int);
			int intSize2 = intSize * 2;
			int[] matrixSize = new int[2];
			double[] data = new double[(bytes.Length - intSize2) / ItemSize];
			Buffer.BlockCopy(bytes, 0, matrixSize, 0, intSize2);
			Buffer.BlockCopy(bytes, intSize2, data, 0, bytes.Length - intSize2);

			int rows = matrixSize[0];
			int cols = matrixSize[1];
			CudaMatrixDouble matrix = new CudaMatrixDouble(rows, cols, false);
			matrix._Host = data;
			matrix._Dirty = true;
			matrix.UpdateDeviceMemory();
			return matrix;
		}

		public byte[] ToByteArray() {
			int intSize = sizeof(int);
			int intSize2 = intSize * 2;
			int size = intSize2 + _Host.Length * ItemSize;
			byte[] bytes = new byte[size];
			Buffer.BlockCopy(new int[] { _Rows, _Cols }, 0, bytes, 0, intSize2);
			Buffer.BlockCopy(_Host, 0, bytes, intSize2, bytes.Length - intSize2);
			return bytes;
		}

		public CudaMatrixDouble Add(CudaMatrixDouble matrix) {
			int rows = Math.Max(_Rows, matrix._Rows);
			int cols = Math.Max(_Cols, matrix._Cols);
			CudaMatrixDouble result = new CudaMatrixDouble(rows, cols);

			UpdateDeviceMemory();
			matrix.UpdateDeviceMemory();

			IntPtr handle = cuBLAS.Create_v2();
			cuBLAS.Dgeam(
				handle,
				cublasOperation.CUBLAS_OP_N,
				cublasOperation.CUBLAS_OP_N,
				_Rows, _Cols,
				1d,
				_Device, _Rows,
				1d,
				matrix._Device, _Rows,
				result._Device, _Rows
			);
			cuBLAS.Destroy_v2(handle);

			result.UpdateHostMemory();
			return result;
		}

		public CudaMatrixDouble Sub(CudaMatrixDouble matrix) {
			int rows = Math.Max(_Rows, matrix._Rows);
			int cols = Math.Max(_Cols, matrix._Cols);
			CudaMatrixDouble result = new CudaMatrixDouble(rows, cols);

			UpdateDeviceMemory();
			matrix.UpdateDeviceMemory();

			IntPtr handle = cuBLAS.Create_v2();
			cuBLAS.Dgeam(
				handle,
				cublasOperation.CUBLAS_OP_N,
				cublasOperation.CUBLAS_OP_N,
				_Rows, _Cols,
				1d,
				_Device, _Rows,
				-1d,
				matrix._Device, _Rows,
				result._Device, _Rows
			);
			cuBLAS.Destroy_v2(handle);

			result.UpdateHostMemory();
			return result;
		}

		public CudaMatrixDouble Mul(CudaMatrixDouble matrix) {
			int rows = Math.Max(_Rows, matrix._Rows);
			int cols = Math.Max(_Cols, matrix._Cols);
			CudaMatrixDouble result = new CudaMatrixDouble(_Rows, matrix._Cols);

			UpdateDeviceMemory();
			matrix.UpdateDeviceMemory();

			IntPtr handle = cuBLAS.Create_v2();
			cuBLAS.Ddgmm(
				handle,
				cublasSideMode.CUBLAS_SIDE_LEFT,
				_Rows, _Cols,
				_Device, _Rows,
				matrix._Device, _Rows,
				result._Device, _Rows
			);
			cuBLAS.Destroy_v2(handle);

			result.UpdateHostMemory();
			return result;
		}

		public CudaMatrixDouble Dot(CudaMatrixDouble matrix) {
			CudaMatrixDouble result = new CudaMatrixDouble(_Rows, matrix._Cols);

			UpdateDeviceMemory();
			matrix.UpdateDeviceMemory();

			IntPtr handle = cuBLAS.Create_v2();
			cuBLAS.Dgemm_v2(
				handle,
				cublasOperation.CUBLAS_OP_N,
				cublasOperation.CUBLAS_OP_N,
				_Rows, matrix._Cols, _Cols,
				1f,
				_Device, _Rows,
				matrix._Device, matrix._Rows,
				1f,
				result._Device, _Rows
			);
			cuBLAS.Destroy_v2(handle);

			result.UpdateHostMemory();
			return result;
		}

		public CudaMatrixDouble Mul(double value) {
			CudaMatrixDouble result = new CudaMatrixDouble(_Rows, _Cols);
			UpdateDeviceMemory();

			IntPtr handle = cuBLAS.Create_v2();
			cuBLAS.Dgeam(
				handle,
				cublasOperation.CUBLAS_OP_N,
				cublasOperation.CUBLAS_OP_N,
				_Rows, _Cols,
				value,
				_Device, _Rows,
				0f,
				IntPtr.Zero, _Rows,
				result._Device, _Rows
			);
			cuBLAS.Destroy_v2(handle);

			result.UpdateHostMemory();
			return result;
		}

		public double Sum() {
			UpdateDeviceMemory();
			IntPtr handle = cuBLAS.Create_v2();
			double result = cuBLAS.Dasum_v2(
				handle,
				Count,
				_Device, 1
			);
			cuBLAS.Destroy_v2(handle);
			return result;
		}

		public int MaxIndex() {
			UpdateDeviceMemory();
			IntPtr handle = cuBLAS.Create_v2();
			int index = cuBLAS.Idamax_v2(
				handle,
				Count,
				_Device, 1
			);
			cuBLAS.Destroy_v2(handle);
			return index - 1;
		}

		public int MinIndex() {
			UpdateDeviceMemory();
			IntPtr handle = cuBLAS.Create_v2();
			int index = cuBLAS.Idamin_v2(
				handle,
				Count,
				_Device, 1
			);
			cuBLAS.Destroy_v2(handle);
			return index - 1;
		}

		public void ForEach(ForEachAction<int, int, double> action) {
			for (int y = 0; y < _Rows; y++) {
				for (int x = 0; x < _Cols; x++) {
					action(x, y, this[x, y]);
				}
			}
		}

		void UpdateHostMemory() {
			_Host = cuBLAS.GetMatrix<double>(_Rows, _Cols, _Device);
		}

		void UpdateDeviceMemory() {
			if (_Dirty == false) {
				return;
			}
			_Dirty = false;
			cuBLAS.SetMatrix<double>(_Rows, _Cols, _Host, _Device);
		}
	}
}
