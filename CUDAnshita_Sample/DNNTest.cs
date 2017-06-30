using System;
using CUDAnshita;

namespace CUDAnshita_Sample {
	class DNNTest {

		public DNNTest() {

		}

		public void Test() {
			IntPtr pInputDesc = IntPtr.Zero;
			IntPtr pOutputDesc = IntPtr.Zero;
			IntPtr pFilterDesc = IntPtr.Zero;
			IntPtr pConvDesc = IntPtr.Zero;

			cudnnDataType dataType = cudnnDataType.CUDNN_DATA_FLOAT;
			int n_in = 64;  // Number of images - originally 128
			int c_in = 96;  // Number of feature maps per image - originally 96
			int h_in = 221; // Height of each feature map - originally 221
			int w_in = 221; // Width of each feature map - originally 221

			cuDNN6 dnn = new cuDNN6();
			pInputDesc = dnn.CreateTensorDescriptor();
			pOutputDesc = dnn.CreateTensorDescriptor();
			pFilterDesc = dnn.CreateFilterDescriptor();
			pConvDesc = dnn.CreateConvolutionDescriptor();

			dnn.SetTensor4dDescriptor(pInputDesc, cudnnTensorFormat.CUDNN_TENSOR_NCHW, dataType, n_in, c_in, h_in, w_in);
			//dnn.Getten

			Console.WriteLine(pInputDesc);
			dnn.DestroyConvolutionDescriptor(pConvDesc);
			dnn.DestroyFilterDescriptor(pFilterDesc);
			dnn.DestroyTensorDescriptor(pOutputDesc);
			dnn.DestroyTensorDescriptor(pInputDesc);
		}
	}
}
