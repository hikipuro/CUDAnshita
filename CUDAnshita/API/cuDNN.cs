using System;
using System.Runtime.InteropServices;

namespace CUDAnshita {
	using cudnnHandle_t = IntPtr;
	using cudnnStatus_t = cudnnStatus;
	using cudaStream_t = IntPtr;
	using cudnnTensorDescriptor_t = IntPtr;
	using cudnnTensorFormat_t = cudnnTensorFormat;
	using cudnnDataType_t = cudnnDataType;
	using cudnnOpTensorDescriptor_t = IntPtr;
	using cudnnOpTensorOp_t = cudnnOpTensorOp;
	using cudnnNanPropagation_t = cudnnNanPropagation;
	using cudnnReduceTensorDescriptor_t = IntPtr;
	using cudnnReduceTensorOp_t = cudnnReduceTensorOp;
	using cudnnReduceTensorIndices_t = cudnnReduceTensorIndices;
	using cudnnIndicesType_t = cudnnIndicesType;
	using cudnnFilterDescriptor_t = IntPtr;
	using cudnnConvolutionDescriptor_t = IntPtr;
	using cudnnConvolutionMode_t = cudnnConvolutionMode;
	using cudnnConvolutionFwdAlgoPerf_t = cudnnConvolutionFwdAlgoPerf;
	using cudnnConvolutionFwdPreference_t = cudnnConvolutionFwdPreference;
	using cudnnConvolutionFwdAlgo_t = cudnnConvolutionFwdAlgo;
	using cudnnActivationDescriptor_t = IntPtr;
	using cudnnConvolutionBwdFilterAlgoPerf_t = cudnnConvolutionBwdFilterAlgoPerf;
	using cudnnConvolutionBwdFilterPreference_t = cudnnConvolutionBwdFilterPreference;
	using cudnnConvolutionBwdFilterAlgo_t = cudnnConvolutionBwdFilterAlgo;
	using cudnnConvolutionBwdDataAlgoPerf_t = cudnnConvolutionBwdDataAlgoPerf;
	using cudnnConvolutionBwdDataPreference_t = cudnnConvolutionBwdDataPreference;
	using cudnnConvolutionBwdDataAlgo_t = cudnnConvolutionBwdDataAlgo;
	using cudnnSoftmaxAlgorithm_t = cudnnSoftmaxAlgorithm;
	using cudnnSoftmaxMode_t = cudnnSoftmaxMode;
	using cudnnPoolingDescriptor_t = IntPtr;
	using cudnnPoolingMode_t = cudnnPoolingMode;
	using cudnnActivationMode_t = cudnnActivationMode;
	using cudnnLRNDescriptor_t = IntPtr;
	using cudnnLRNMode_t = cudnnLRNMode;
	using cudnnDivNormMode_t = cudnnDivNormMode;
	using cudnnBatchNormMode_t = cudnnBatchNormMode;
	using cudnnSpatialTransformerDescriptor_t = IntPtr;
	using cudnnSamplerType_t = cudnnSamplerType;
	using cudnnDropoutDescriptor_t = IntPtr;
	using cudnnRNNDescriptor_t = IntPtr;
	using cudnnPersistentRNNPlan_t = IntPtr;
	using cudnnRNNInputMode_t = cudnnRNNInputMode;
	using cudnnDirectionMode_t = cudnnDirectionMode;
	using cudnnRNNMode_t = cudnnRNNMode;
	using cudnnRNNAlgo_t = cudnnRNNAlgo;
	using size_t = Int64;
	using System.Text;

	/// <summary>
	/// Neural Networks Library.
	/// </summary>
	/// <remarks>
	/// <a href="https://developer.nvidia.com/cudnn">https://developer.nvidia.com/cudnn</a>
	/// </remarks>
	public class cuDNN6 : IDisposable {
		/// <summary>
		/// cuDNN6 DLL functions.
		/// </summary>
		public class API {
			const string DLL_PATH = "cudnn64_6.dll";
			const CallingConvention CALLING_CONVENTION = CallingConvention.Cdecl;
			const CharSet CHAR_SET = CharSet.Ansi;

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern size_t cudnnGetVersion();

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern size_t cudnnGetCudartVersion();

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern IntPtr cudnnGetErrorString(cudnnStatus_t status);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetProperty(libraryPropertyType type, ref int value);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnCreate(ref cudnnHandle_t handle);
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId);
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, ref cudaStream_t streamId);


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnCreateTensorDescriptor(
											ref cudnnTensorDescriptor_t tensorDesc);


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetTensor4dDescriptor(
											cudnnTensorDescriptor_t tensorDesc,
											cudnnTensorFormat_t format,
											cudnnDataType_t dataType, // image data type
											int n,        // number of inputs (batch size)
											int c,        // number of input feature maps
											int h,        // height of input section
											int w);       // width of input section


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetTensor4dDescriptorEx(
											cudnnTensorDescriptor_t tensorDesc,
											cudnnDataType_t dataType, // image data type
											int n,        // number of inputs (batch size)
											int c,        // number of input feature maps
											int h,        // height of input section
											int w,        // width of input section
											int nStride,
											int cStride,
											int hStride,
											int wStride);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetTensor4dDescriptor(
								cudnnTensorDescriptor_t tensorDesc,
								ref cudnnDataType_t dataType, // image data type
								ref int n,        // number of inputs (batch size)
								ref int c,        // number of input feature maps
								ref int h,        // height of input section
								ref int w,        // width of input section
								ref int nStride,
								ref int cStride,
								ref int hStride,
								ref int wStride);


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetTensorNdDescriptor(
											cudnnTensorDescriptor_t tensorDesc,
											cudnnDataType_t dataType,
											int nbDims,
											int[] dimA,
											int[] strideA);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetTensorNdDescriptorEx(
											cudnnTensorDescriptor_t tensorDesc,
											cudnnTensorFormat_t format,
											cudnnDataType_t dataType,
											int nbDims,
											int[] dimA);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetTensorNdDescriptor(
								cudnnTensorDescriptor_t tensorDesc,
								int nbDimsRequested,
								ref cudnnDataType_t dataType,
								ref int nbDims,
								int[] dimA,
								int[] strideA);


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetTensorSizeInBytes(
								cudnnTensorDescriptor_t tensorDesc,
								ref size_t size);


			// Destroy an instance of Tensor4d descriptor
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDestroyTensorDescriptor(
											cudnnTensorDescriptor_t tensorDesc);


			// Tensor layout conversion helper (y = alpha * x + beta * y)
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnTransformTensor(
											cudnnHandle_t handle,
											IntPtr alpha,
											cudnnTensorDescriptor_t xDesc,
											IntPtr x,
											IntPtr beta,
											cudnnTensorDescriptor_t yDesc,
											IntPtr y);


			// Tensor Bias addition : C = alpha * A + beta * C 
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnAddTensor(
											cudnnHandle_t handle,
											IntPtr alpha,
											cudnnTensorDescriptor_t aDesc,
											IntPtr A,
											IntPtr beta,
											cudnnTensorDescriptor_t cDesc,
											IntPtr C );


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnCreateOpTensorDescriptor(
											ref cudnnOpTensorDescriptor_t opTensorDesc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetOpTensorDescriptor(
											cudnnOpTensorDescriptor_t opTensorDesc,
											cudnnOpTensorOp_t opTensorOp,
											cudnnDataType_t opTensorCompType,
											cudnnNanPropagation_t opTensorNanOpt);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetOpTensorDescriptor(
								cudnnOpTensorDescriptor_t opTensorDesc,
								ref cudnnOpTensorOp_t opTensorOp,
								ref cudnnDataType_t opTensorCompType,
								ref cudnnNanPropagation_t opTensorNanOpt);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDestroyOpTensorDescriptor(
								cudnnOpTensorDescriptor_t opTensorDesc);

			// Tensor operation : C = op( alpha1 * A, alpha2 * B ) + beta * C
			// B tensor is ignored for CUDNN_OP_TENSOR_SQRT.
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnOpTensor(
											cudnnHandle_t handle,
											cudnnOpTensorDescriptor_t opTensorDesc,
											IntPtr alpha1,
											cudnnTensorDescriptor_t aDesc,
											IntPtr A,
											IntPtr alpha2,
											cudnnTensorDescriptor_t bDesc,
											IntPtr B,
											IntPtr beta,
											cudnnTensorDescriptor_t cDesc,
											IntPtr C );


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnCreateReduceTensorDescriptor(
											ref cudnnReduceTensorDescriptor_t reduceTensorDesc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetReduceTensorDescriptor(
											cudnnReduceTensorDescriptor_t reduceTensorDesc,
											cudnnReduceTensorOp_t reduceTensorOp,
											cudnnDataType_t reduceTensorCompType,
											cudnnNanPropagation_t reduceTensorNanOpt,
											cudnnReduceTensorIndices_t reduceTensorIndices,
											cudnnIndicesType_t reduceTensorIndicesType);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetReduceTensorDescriptor(
								cudnnReduceTensorDescriptor_t reduceTensorDesc,
								ref cudnnReduceTensorOp_t reduceTensorOp,
								ref cudnnDataType_t reduceTensorCompType,
								ref cudnnNanPropagation_t reduceTensorNanOpt,
								ref cudnnReduceTensorIndices_t reduceTensorIndices,
								ref cudnnIndicesType_t reduceTensorIndicesType);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDestroyReduceTensorDescriptor(
								cudnnReduceTensorDescriptor_t reduceTensorDesc);

			// Helper function to return the minimum size of the index space to be passed to the reduction given the input and output tensors
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetReductionIndicesSize(
											cudnnHandle_t handle,
											cudnnReduceTensorDescriptor_t reduceTensorDesc,
											cudnnTensorDescriptor_t aDesc,
											cudnnTensorDescriptor_t cDesc,
											ref size_t sizeInBytes );

			// Helper function to return the minimum size of the workspace to be passed to the reduction given the input and output tensors
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetReductionWorkspaceSize(
								cudnnHandle_t handle,
								cudnnReduceTensorDescriptor_t reduceTensorDesc,
								cudnnTensorDescriptor_t aDesc,
								cudnnTensorDescriptor_t cDesc,
								ref size_t sizeInBytes);

			// Tensor operation : C = reduce op( alpha * A ) + beta * C
			// The NaN propagation enum applies to only the min and max reduce ops; the other reduce ops propagate NaN as usual.
			// The indices space is ignored for reduce ops other than min or max.
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnReduceTensor(
								cudnnHandle_t handle,
								cudnnReduceTensorDescriptor_t reduceTensorDesc,
								IntPtr indices,
								size_t indicesSizeInBytes,
								IntPtr workspace,
								size_t workspaceSizeInBytes,
								IntPtr alpha,
								cudnnTensorDescriptor_t aDesc,
								IntPtr A,
								IntPtr beta,
								cudnnTensorDescriptor_t cDesc,
								IntPtr C);

			// Set all values of a tensor to a given value : y[i] = value[0]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetTensor(
											cudnnHandle_t handle,
											cudnnTensorDescriptor_t yDesc,
											IntPtr y,
											IntPtr valuePtr );

			// Scale all values of a tensor by a given factor : y[i] = alpha * y[i]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnScaleTensor(
											cudnnHandle_t handle,
											cudnnTensorDescriptor_t yDesc,
											IntPtr y,
											IntPtr alpha);


			// Create an instance of FilterStruct
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnCreateFilterDescriptor(
											ref cudnnFilterDescriptor_t filterDesc);


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetFilter4dDescriptor(
											cudnnFilterDescriptor_t filterDesc,
											cudnnDataType_t dataType, // image data type
											cudnnTensorFormat_t format,
											int k,        // number of output feature maps
											int c,        // number of input feature maps
											int h,        // height of each input filter
											int w);      // width of  each input filter


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetFilter4dDescriptor(
								cudnnFilterDescriptor_t filterDesc,
								ref cudnnDataType_t                    dataType, // image data type
                                ref cudnnTensorFormat_t format,
								ref int k,        // number of output feature maps
								ref int c,        // number of input feature maps
								ref int h,        // height of each input filter
								ref int w );      // width of  each input filter


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetFilterNdDescriptor(
											cudnnFilterDescriptor_t filterDesc,
											cudnnDataType_t dataType, // image data type
											cudnnTensorFormat_t format,
											int nbDims,
											int[] filterDimA );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetFilterNdDescriptor(
								cudnnFilterDescriptor_t filterDesc,
								int nbDimsRequested,
								ref cudnnDataType_t dataType, // image data type
								ref cudnnTensorFormat_t format,
								ref int nbDims,
								int[] filterDimA);


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDestroyFilterDescriptor(
											cudnnFilterDescriptor_t filterDesc);

			// Create an instance of convolution descriptor
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnCreateConvolutionDescriptor(
											ref cudnnConvolutionDescriptor_t convDesc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc,
																		 int pad_h,    // zero-padding height
																		 int pad_w,    // zero-padding width
																		 int u,   // vertical filter stride
																		 int v,   // horizontal filter stride
																		 int dilation_h, // filter dilation in the vertical dimension
																		 int dilation_w, // filter dilation in the horizontal dimension
																		 cudnnConvolutionMode_t mode,
																		 cudnnDataType_t computeType
																	   );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc,
															ref int pad_h,    // zero-padding height
															ref int pad_w,    // zero-padding width
															ref int u,        // vertical filter stride
															ref int v,        // horizontal filter stride
															ref int dilation_h, // filter dilation in the vertical dimension
															ref int dilation_w, // filter dilation in the horizontal dimension
															ref cudnnConvolutionMode_t mode,
                                                            ref cudnnDataType_t computeType
                                                         );

			// Helper function to return the dimensions of the output tensor given a convolution descriptor
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(
								cudnnConvolutionDescriptor_t convDesc,
								cudnnTensorDescriptor_t inputTensorDesc,
								cudnnFilterDescriptor_t filterDesc,
								ref int n,
								ref int c,
								ref int h,
								ref int w );


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetConvolutionNdDescriptor(
											cudnnConvolutionDescriptor_t convDesc,
											int arrayLength,             // nbDims-2 size
											int[] padA,
											int[] filterStrideA,
											int[] dilationA,
											cudnnConvolutionMode_t              mode,
											cudnnDataType_t computeType );  // convolution data type

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetConvolutionNdDescriptor(
								cudnnConvolutionDescriptor_t convDesc,
								int arrayLengthRequested,
								ref int arrayLength,
								int[] padA,
								int[] strideA,
								int[] dilationA,
								ref cudnnConvolutionMode_t             mode,
                                ref cudnnDataType_t computeType );   // convolution data type


			// Helper function to return the dimensions of the output tensor given a convolution descriptor
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim(
								cudnnConvolutionDescriptor_t convDesc,
								cudnnTensorDescriptor_t inputTensorDesc,
								cudnnFilterDescriptor_t filterDesc,
								int nbDims,
								int[] tensorOuputDimA);

			// Destroy an instance of convolution descriptor
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDestroyConvolutionDescriptor(
											cudnnConvolutionDescriptor_t convDesc);


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(
											cudnnHandle_t handle,
								 cudnnTensorDescriptor_t xDesc,
								 cudnnFilterDescriptor_t wDesc,
								 cudnnConvolutionDescriptor_t convDesc,
								 cudnnTensorDescriptor_t yDesc,
								 int requestedAlgoCount,
								ref int returnedAlgoCount,
								ref cudnnConvolutionFwdAlgoPerf_t      perfResults );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx(
								cudnnHandle_t handle,
								 cudnnTensorDescriptor_t xDesc,
								 IntPtr x,
								 cudnnFilterDescriptor_t wDesc,
								 IntPtr w,
								 cudnnConvolutionDescriptor_t convDesc,
								 cudnnTensorDescriptor_t yDesc,
								IntPtr y,
								 int requestedAlgoCount,
								ref int returnedAlgoCount,
								ref cudnnConvolutionFwdAlgoPerf_t perfResults,
								IntPtr workSpace,
								size_t workSpaceSizeInBytes);


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetConvolutionForwardAlgorithm(
								cudnnHandle_t handle,
								 cudnnTensorDescriptor_t xDesc,
								 cudnnFilterDescriptor_t wDesc,
								 cudnnConvolutionDescriptor_t convDesc,
								 cudnnTensorDescriptor_t yDesc,
								cudnnConvolutionFwdPreference_t     preference,
                                size_t memoryLimitInBytes,
								ref cudnnConvolutionFwdAlgo_t          algo );

			//
			// convolution algorithm (which requires potentially some workspace)
			//

			// Helper function to return the minimum size of the workspace to be passed to the convolution given an algo
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
								cudnnHandle_t handle,
								 cudnnTensorDescriptor_t xDesc,
								 cudnnFilterDescriptor_t wDesc,
								 cudnnConvolutionDescriptor_t convDesc,
								 cudnnTensorDescriptor_t yDesc,
								cudnnConvolutionFwdAlgo_t           algo,
                                ref size_t sizeInBytes );


			// Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output"

			// Function to perform the forward pass for batch convolution
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnConvolutionForward(
											cudnnHandle_t handle,
								 IntPtr alpha,
								 cudnnTensorDescriptor_t xDesc,
								 IntPtr x,
								 cudnnFilterDescriptor_t wDesc,
								 IntPtr w,
								 cudnnConvolutionDescriptor_t convDesc,
								cudnnConvolutionFwdAlgo_t algo,
								IntPtr workSpace,
								size_t workSpaceSizeInBytes,
								IntPtr beta,
								 cudnnTensorDescriptor_t yDesc,
								IntPtr y);

			// Fused conv/bias/activation operation : y = Act( alpha1 * conv(x) + alpha2 * z + bias )
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnConvolutionBiasActivationForward(
											cudnnHandle_t handle,
								 IntPtr alpha1,
								 cudnnTensorDescriptor_t xDesc,
								 IntPtr x,
								 cudnnFilterDescriptor_t wDesc,
								 IntPtr w,
								 cudnnConvolutionDescriptor_t convDesc,
								cudnnConvolutionFwdAlgo_t           algo,
                                IntPtr workSpace,
								size_t                              workSpaceSizeInBytes,
                                IntPtr alpha2,
								 cudnnTensorDescriptor_t zDesc,
								 IntPtr z,
								 cudnnTensorDescriptor_t biasDesc,
								 IntPtr bias,
								 cudnnActivationDescriptor_t activationDesc,
								 cudnnTensorDescriptor_t yDesc,
								IntPtr y );

			// Function to compute the bias gradient for batch convolution
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnConvolutionBackwardBias(
											cudnnHandle_t handle,
								 IntPtr alpha,
								 cudnnTensorDescriptor_t dyDesc,
								 IntPtr dy,
								 IntPtr beta,
								 cudnnTensorDescriptor_t dbDesc,
								IntPtr db );


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(
											cudnnHandle_t handle,
											 cudnnTensorDescriptor_t xDesc,
											 cudnnTensorDescriptor_t dyDesc,
											 cudnnConvolutionDescriptor_t convDesc,
											 cudnnFilterDescriptor_t dwDesc,
											 int requestedAlgoCount,
											ref int returnedAlgoCount,
											ref cudnnConvolutionBwdFilterAlgoPerf_t perfResults );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx(
								cudnnHandle_t handle,
								 cudnnTensorDescriptor_t xDesc,
								 IntPtr x,
								 cudnnTensorDescriptor_t dyDesc,
								 IntPtr y,
								 cudnnConvolutionDescriptor_t convDesc,
								 cudnnFilterDescriptor_t dwDesc,
								IntPtr dw,
								 int requestedAlgoCount,
								ref int returnedAlgoCount,
								ref cudnnConvolutionBwdFilterAlgoPerf_t perfResults,
								IntPtr workSpace,
								size_t workSpaceSizeInBytes);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(
								cudnnHandle_t handle,
								 cudnnTensorDescriptor_t xDesc,
								 cudnnTensorDescriptor_t dyDesc,
								 cudnnConvolutionDescriptor_t convDesc,
								 cudnnFilterDescriptor_t dwDesc,
								cudnnConvolutionBwdFilterPreference_t preference,
								size_t memoryLimitInBytes,
								ref cudnnConvolutionBwdFilterAlgo_t algo);

			//
			// convolution algorithm (which requires potentially some workspace)
			//

			// Helper function to return the minimum size of the workspace to be passed to the convolution given an algo
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
								cudnnHandle_t handle,
								 cudnnTensorDescriptor_t xDesc,
								 cudnnTensorDescriptor_t dyDesc,
								 cudnnConvolutionDescriptor_t convDesc,
								 cudnnFilterDescriptor_t gradDesc,
								cudnnConvolutionBwdFilterAlgo_t     algo,
                                ref size_t sizeInBytes );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnConvolutionBackwardFilter(
											cudnnHandle_t handle,
								 IntPtr alpha,
								 cudnnTensorDescriptor_t xDesc,
								 IntPtr x,
								 cudnnTensorDescriptor_t dyDesc,
								 IntPtr dy,
								 cudnnConvolutionDescriptor_t convDesc,
								cudnnConvolutionBwdFilterAlgo_t     algo,
                                IntPtr workSpace,
								size_t                              workSpaceSizeInBytes,
                                IntPtr beta,
								 cudnnFilterDescriptor_t dwDesc,
								IntPtr dw );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(
											cudnnHandle_t handle,
								 cudnnFilterDescriptor_t wDesc,
								 cudnnTensorDescriptor_t dyDesc,
								 cudnnConvolutionDescriptor_t convDesc,
								 cudnnTensorDescriptor_t dxDesc,
								 int requestedAlgoCount,
								ref int returnedAlgoCount,
								ref cudnnConvolutionBwdDataAlgoPerf_t  perfResults );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx(
								cudnnHandle_t handle,
								 cudnnFilterDescriptor_t wDesc,
								 IntPtr w,
								 cudnnTensorDescriptor_t dyDesc,
								 IntPtr dy,
								 cudnnConvolutionDescriptor_t convDesc,
								 cudnnTensorDescriptor_t dxDesc,
								IntPtr dx,
								 int requestedAlgoCount,
								ref int returnedAlgoCount,
								ref cudnnConvolutionBwdDataAlgoPerf_t  perfResults,
                                IntPtr workSpace,
								size_t                              workSpaceSizeInBytes );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm(
								cudnnHandle_t handle,
								 cudnnFilterDescriptor_t wDesc,
								 cudnnTensorDescriptor_t dyDesc,
								 cudnnConvolutionDescriptor_t convDesc,
								 cudnnTensorDescriptor_t dxDesc,
								cudnnConvolutionBwdDataPreference_t preference,
								size_t memoryLimitInBytes,
								ref cudnnConvolutionBwdDataAlgo_t algo);

			// Helper function to return the minimum size of the workspace to be passed to the convolution given an algo
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
								cudnnHandle_t handle,
								 cudnnFilterDescriptor_t wDesc,
								 cudnnTensorDescriptor_t dyDesc,
								 cudnnConvolutionDescriptor_t convDesc,
								 cudnnTensorDescriptor_t dxDesc,
								cudnnConvolutionBwdDataAlgo_t       algo,
                                ref size_t sizeInBytes );


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnConvolutionBackwardData(
											cudnnHandle_t handle,
								 IntPtr alpha,
								 cudnnFilterDescriptor_t wDesc,
								 IntPtr w,
								 cudnnTensorDescriptor_t dyDesc,
								 IntPtr dy,
								 cudnnConvolutionDescriptor_t convDesc,
								cudnnConvolutionBwdDataAlgo_t       algo,
                                IntPtr workSpace,
								size_t                              workSpaceSizeInBytes,
                                IntPtr beta,
								 cudnnTensorDescriptor_t dxDesc,
								IntPtr dx );


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnIm2Col(
											cudnnHandle_t handle,
								 cudnnTensorDescriptor_t xDesc,
								 IntPtr x,
								 cudnnFilterDescriptor_t wDesc,
								 cudnnConvolutionDescriptor_t convDesc,
								IntPtr colBuffer );

			// Function to perform forward softmax
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSoftmaxForward(
											cudnnHandle_t handle,
											cudnnSoftmaxAlgorithm_t algo,
											cudnnSoftmaxMode_t mode,
								 IntPtr alpha,
								 cudnnTensorDescriptor_t xDesc,
								 IntPtr x,
								 IntPtr beta,
								 cudnnTensorDescriptor_t yDesc,
								IntPtr y );

			// Function to perform backward softmax
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSoftmaxBackward(
											cudnnHandle_t handle,
											cudnnSoftmaxAlgorithm_t algo,
											cudnnSoftmaxMode_t mode,
								 IntPtr alpha,
								 cudnnTensorDescriptor_t yDesc,
								 IntPtr y,
								 cudnnTensorDescriptor_t dyDesc,
								 IntPtr dy,
								 IntPtr beta,
								 cudnnTensorDescriptor_t dxDesc,
								IntPtr dx );


			// Create an instance of pooling descriptor
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnCreatePoolingDescriptor(
											ref cudnnPoolingDescriptor_t poolingDesc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetPooling2dDescriptor(
											cudnnPoolingDescriptor_t poolingDesc,
											cudnnPoolingMode_t mode,
											cudnnNanPropagation_t maxpoolingNanOpt,
											int windowHeight,
											int windowWidth,
											int verticalPadding,
											int horizontalPadding,
											int verticalStride,
											int horizontalStride);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetPooling2dDescriptor(
								 cudnnPoolingDescriptor_t poolingDesc,
								ref cudnnPoolingMode_t mode,
								ref cudnnNanPropagation_t maxpoolingNanOpt,
								ref int windowHeight,
								ref int windowWidth,
								ref int verticalPadding,
								ref int horizontalPadding,
								ref int verticalStride,
								ref int horizontalStride);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetPoolingNdDescriptor(
											cudnnPoolingDescriptor_t poolingDesc,
								 cudnnPoolingMode_t mode,
								 cudnnNanPropagation_t maxpoolingNanOpt,
								int nbDims,
								 int[] windowDimA,
								 int[] paddingA,
								 int[] strideA);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetPoolingNdDescriptor(
								 cudnnPoolingDescriptor_t poolingDesc,
								int nbDimsRequested,
								ref cudnnPoolingMode_t mode,
								ref cudnnNanPropagation_t maxpoolingNanOpt,
								ref int nbDims,
								int[] windowDimA,
								int[] paddingA,
								int[] strideA);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetPoolingNdForwardOutputDim(
								 cudnnPoolingDescriptor_t poolingDesc,
								 cudnnTensorDescriptor_t inputTensorDesc,
								int nbDims,
								int[] outputTensorDimA);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetPooling2dForwardOutputDim(
								 cudnnPoolingDescriptor_t poolingDesc,
								 cudnnTensorDescriptor_t inputTensorDesc,
								ref int n,
								ref int c,
								ref int h,
								ref int w);


			// Destroy an instance of pooling descriptor
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDestroyPoolingDescriptor(
											cudnnPoolingDescriptor_t poolingDesc);

			// Pooling functions: All of the form "output = alpha * Op(inputs) + beta * output"

			// Function to perform forward pooling
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnPoolingForward(
											cudnnHandle_t handle,
								 cudnnPoolingDescriptor_t poolingDesc,
								 IntPtr alpha,
								 cudnnTensorDescriptor_t xDesc,
								 IntPtr x,
								 IntPtr beta,
								 cudnnTensorDescriptor_t yDesc,
								IntPtr y );

			// Function to perform backward pooling
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnPoolingBackward(
											cudnnHandle_t handle,
								 cudnnPoolingDescriptor_t poolingDesc,
								 IntPtr alpha,
								 cudnnTensorDescriptor_t yDesc,
								 IntPtr y,
								 cudnnTensorDescriptor_t dyDesc,
								 IntPtr dy,
								 cudnnTensorDescriptor_t xDesc,
								 IntPtr x,
								 IntPtr beta,
								 cudnnTensorDescriptor_t dxDesc,
								IntPtr dx );


			// Activation functions: All of the form "output = alpha * Op(inputs) + beta * output"
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnCreateActivationDescriptor(
											ref cudnnActivationDescriptor_t activationDesc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetActivationDescriptor(
											cudnnActivationDescriptor_t activationDesc,
											cudnnActivationMode_t mode,
											cudnnNanPropagation_t reluNanOpt,
											double coef); // ceiling for clipped RELU, alpha for ELU

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetActivationDescriptor(
								 cudnnActivationDescriptor_t activationDesc,
								ref cudnnActivationMode_t              mode,
                                ref cudnnNanPropagation_t reluNanOpt,
								ref double coef ); // ceiling for clipped RELU, alpha for ELU

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDestroyActivationDescriptor(
											cudnnActivationDescriptor_t activationDesc);

			// Function to perform forward activation 
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnActivationForward(
											cudnnHandle_t handle,
											cudnnActivationDescriptor_t activationDesc,
								 IntPtr alpha,
								 cudnnTensorDescriptor_t xDesc,
								 IntPtr x,
								 IntPtr beta,
								 cudnnTensorDescriptor_t yDesc,
								IntPtr y );

			// Function to perform backward activation 
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnActivationBackward(
											cudnnHandle_t handle,
											cudnnActivationDescriptor_t activationDesc,
								 IntPtr alpha,
								 cudnnTensorDescriptor_t yDesc,
								 IntPtr y,
								 cudnnTensorDescriptor_t dyDesc,
								 IntPtr dy,
								 cudnnTensorDescriptor_t xDesc,
								 IntPtr x,
								 IntPtr beta,
								 cudnnTensorDescriptor_t dxDesc,
								IntPtr dx );

			// 
			// Create an instance of LRN (Local Response Normalization) descriptor
			// Uses lrnN=5, lrnAlpha=1e-4, lrnBeta=0.75, lrnK=2.0 as defaults from Krizhevsky'12 ImageNet paper
			//
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnCreateLRNDescriptor(
											ref cudnnLRNDescriptor_t normDesc);

			//
			// Uses a window [center-lookBehind, center+lookAhead], where
			// lookBehind = floor( (lrnN-1)/2 ), lookAhead = lrnN-lookBehind-1.
			// Values of double parameters cast to tensor data type.
			//
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetLRNDescriptor(
											cudnnLRNDescriptor_t normDesc,
											uint lrnN,
											double lrnAlpha,
											double lrnBeta,
											double lrnK);
			//
			// Retrieve the settings currently stored in an LRN layer descriptor
			// Any of the provided pointers can be NULL (no corresponding value will be returned)
			//
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetLRNDescriptor(
											cudnnLRNDescriptor_t normDesc,
											ref uint lrnN,
											ref double lrnAlpha,
											ref double lrnBeta,
											ref double lrnK);

			// Destroy an instance of LRN descriptor
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc);

			// LRN functions: output = alpha * normalize(x) + beta * old_y

			// LRN cross-channel forward computation. Double parameters cast to tensor data type
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnLRNCrossChannelForward(
											cudnnHandle_t handle,
											cudnnLRNDescriptor_t normDesc,
											cudnnLRNMode_t lrnMode,
								 IntPtr alpha,
								 cudnnTensorDescriptor_t xDesc,
								 IntPtr x,
								 IntPtr beta,
								 cudnnTensorDescriptor_t yDesc,
								IntPtr y );

			// LRN cross-channel backward computation. Double parameters cast to tensor data type
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnLRNCrossChannelBackward(
											cudnnHandle_t handle,
											cudnnLRNDescriptor_t normDesc,
											cudnnLRNMode_t lrnMode,
								 IntPtr alpha,
								 cudnnTensorDescriptor_t yDesc,
								 IntPtr y,
								 cudnnTensorDescriptor_t dyDesc,
								 IntPtr dy,
								 cudnnTensorDescriptor_t xDesc,
								 IntPtr x,
								 IntPtr beta,
								 cudnnTensorDescriptor_t dxDesc,
								IntPtr dx);

			// LCN/divisive normalization functions: y = alpha * normalize(x) + beta * y
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDivisiveNormalizationForward(
											cudnnHandle_t handle,
											cudnnLRNDescriptor_t normDesc,
											cudnnDivNormMode_t mode,
								 IntPtr alpha,
								 cudnnTensorDescriptor_t xDesc, // same desc for means, temp, temp2
								 IntPtr x,
								 IntPtr means, // if NULL, means are assumed to be zero
								IntPtr temp,
								IntPtr temp2,
								 IntPtr beta,
								 cudnnTensorDescriptor_t yDesc,
								IntPtr y );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDivisiveNormalizationBackward(
											cudnnHandle_t handle,
											cudnnLRNDescriptor_t normDesc,
											cudnnDivNormMode_t mode,
								 IntPtr alpha,
								 cudnnTensorDescriptor_t xDesc, // same desc for x, means, dy, temp, temp2
								 IntPtr x,
								 IntPtr means, // if NULL, means are assumed to be zero
								 IntPtr dy,
								IntPtr temp,
								IntPtr temp2,
								 IntPtr beta,
								 cudnnTensorDescriptor_t dXdMeansDesc, // same desc for dx, dMeans
								IntPtr dx, // output x differential
								IntPtr dMeans ); // output means differential, can be NULL

			//
			// Derives a tensor descriptor from layer data descriptor for BatchNormalization 
			// scale, invVariance, bnBias, bnScale tensors. Use this tensor desc for 
			// bnScaleBiasMeanVarDesc and bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
			//
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDeriveBNTensorDescriptor(
											cudnnTensorDescriptor_t derivedBnDesc,
								 cudnnTensorDescriptor_t xDesc,
								cudnnBatchNormMode_t                mode );

			// Computes y = BN(x). Also accumulates moving averages of mean and inverse variances
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnBatchNormalizationForwardTraining(
								cudnnHandle_t handle,
								cudnnBatchNormMode_t mode,
								 IntPtr alpha, // alpha[0] = result blend factor
								 IntPtr beta,  // beta[0] = dest layer blend factor
								 cudnnTensorDescriptor_t xDesc,
								 IntPtr x,     // NxCxHxW
								 cudnnTensorDescriptor_t yDesc,
								IntPtr y,     // NxCxHxW

								// Shared desc for the next 6 tensors in the argument list.
								// Data type to be set as follows:
								// type = (typeOf(x) == double) ? double : float
								// Dimensions for this descriptor depend on normalization mode
								// - Spatial Normalization : tensors are expected to have dims 1xCx1x1
								// (normalization is performed across NxHxW)
								// - Per-Activation Normalization : tensors are expected to have dims of 1xCxHxW 
								// (normalization is performed across N)
								// cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,

								// 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation
								IntPtr bnScale,

								 IntPtr bnBias,

								// MUST use factor=1 in the very first call of a complete training cycle.
								// Use a factor=1/(1+n) at N-th call to the function to get
								// Cumulative Moving Average (CMA) behavior
								// CMA[n] = (x[1]+...+x[n])/n
								// Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1) =
								// ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) =
								// CMA[n]*(1-1/(n+1)) + x[n+1]*1/(n+1)
								double exponentialAverageFactor,

								// Used in Training phase only. 
                                // runningMean = newMean*factor + runningMean*(1-factor)
								IntPtr resultRunningMean,
								// Output in training mode, input in inference. Is the moving average
                                // of  variance[x] (factor is applied in the same way as for runningMean)
								IntPtr resultRunningVariance,

								// Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions.
								double epsilon,

								// Optionally save intermediate results from the forward pass here
								//   - can be reused to speed up backward pass. NULL if unused
								IntPtr resultSaveMean,

								IntPtr resultSaveInvVariance );

			//
			// Performs Batch Normalization during Inference: 
			// y[i] = bnScale[k]*(x[i]-estimatedMean[k])/sqrt(epsilon+estimatedVariance[k]) + bnBias[k]
			// with bnScale, bnBias, runningMean, runningInvVariance tensors indexed
			// according to spatial or per-activation mode. Refer to cudnnBatchNormalizationForwardTraining
			// above for notes on function arguments.
			//
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnBatchNormalizationForwardInference(
											cudnnHandle_t handle,
											cudnnBatchNormMode_t mode,
								 IntPtr alpha, // alpha[0] = result blend factor
								 IntPtr beta,  // beta[0] = dest layer blend factor
								 cudnnTensorDescriptor_t xDesc,
								 IntPtr x,     // NxCxHxW
								 cudnnTensorDescriptor_t yDesc,
								IntPtr y,     // NxCxHxW
								 cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
								 IntPtr bnScale,
								 IntPtr bnBias,
								 IntPtr estimatedMean,
								 IntPtr estimatedVariance,
								double epsilon );

			// Performs backward pass of Batch Normalization layer. Returns x gradient,
			// bnScale gradient and bnBias gradient
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnBatchNormalizationBackward(
											cudnnHandle_t handle,
											cudnnBatchNormMode_t mode,
								 IntPtr alphaDataDiff,
								 IntPtr betaDataDiff,
								 IntPtr alphaParamDiff,
								 IntPtr betaParamDiff,
								 cudnnTensorDescriptor_t xDesc, // same desc for x, dx, dy
								 IntPtr x,
								 cudnnTensorDescriptor_t dyDesc,
								 IntPtr dy,
								 cudnnTensorDescriptor_t dxDesc,
								IntPtr dx,
								// Shared tensor desc for the 4 tensors below
								 cudnnTensorDescriptor_t dBnScaleBiasDesc,
								 IntPtr bnScale, // bnBias doesn't affect backpropagation
								// scale and bias diff are not backpropagated below this layer
								IntPtr dBnScaleResult,
								IntPtr dBnBiasResult,
								// Same epsilon as forward pass
								double epsilon,
								// Optionally cached intermediate results from forward pass
								 IntPtr savedMean,
								 IntPtr savedInvVariance );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnCreateSpatialTransformerDescriptor(
										   ref cudnnSpatialTransformerDescriptor_t stDesc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor(
											cudnnSpatialTransformerDescriptor_t stDesc,
											cudnnSamplerType_t samplerType,
											cudnnDataType_t dataType,
								 int nbDims,
								 int[] dimA);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDestroySpatialTransformerDescriptor(
											 cudnnSpatialTransformerDescriptor_t stDesc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSpatialTfGridGeneratorForward(
											 cudnnHandle_t handle,
								 cudnnSpatialTransformerDescriptor_t stDesc,
								 IntPtr theta,
								 IntPtr grid);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSpatialTfGridGeneratorBackward(
											 cudnnHandle_t handle,
								 cudnnSpatialTransformerDescriptor_t stDesc,
								 IntPtr dgrid,
								 IntPtr dtheta);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSpatialTfSamplerForward(
											 cudnnHandle_t handle,
											 cudnnSpatialTransformerDescriptor_t stDesc,
								 IntPtr alpha,
								 cudnnTensorDescriptor_t xDesc,
								 IntPtr x,
								 IntPtr grid,
								 IntPtr beta,
								 cudnnTensorDescriptor_t                    yDesc,
                                 IntPtr y);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSpatialTfSamplerBackward(
											 cudnnHandle_t handle,
											 cudnnSpatialTransformerDescriptor_t stDesc,
								 IntPtr alpha,
								 cudnnTensorDescriptor_t xDesc,
								 IntPtr x,
								 IntPtr beta,
								 cudnnTensorDescriptor_t dxDesc,
								 IntPtr dx,
								 IntPtr alphaDgrid,
								 cudnnTensorDescriptor_t dyDesc,
								 IntPtr dy,
								 IntPtr grid,
								 IntPtr betaDgrid,
								 IntPtr dgrid);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnCreateDropoutDescriptor(ref cudnnDropoutDescriptor_t dropoutDesc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc);

			// helper function to determine size of the states to be passed to cudnnSetDropoutDescriptor
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDropoutGetStatesSize(cudnnHandle_t handle, ref size_t sizeInBytes);

			// helper function to determine size of the reserve space to be passed to dropout forward/backward calls
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xdesc, ref size_t sizeInBytes);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
																cudnnHandle_t handle,
																float dropout,
																IntPtr states,
																size_t stateSizeInBytes,
																ulong seed);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDropoutForward(cudnnHandle_t handle,

													  cudnnDropoutDescriptor_t dropoutDesc,

													  cudnnTensorDescriptor_t xdesc,

													  IntPtr x,

													  cudnnTensorDescriptor_t ydesc,

													  IntPtr y,

													  IntPtr reserveSpace,
													  size_t reserveSpaceSizeInBytes);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDropoutBackward(cudnnHandle_t handle,

											   cudnnDropoutDescriptor_t dropoutDesc,

											   cudnnTensorDescriptor_t dydesc,

											   IntPtr dy,

											   cudnnTensorDescriptor_t dxdesc,

											   IntPtr dx,

											   IntPtr reserveSpace,
											   size_t reserveSpaceSizeInBytes);
				
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnCreateRNNDescriptor(ref cudnnRNNDescriptor_t rnnDesc);
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc);

			// Expensive. Creates the plan for the specific settings.
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc,
											 int minibatch,
											 cudnnDataType_t dataType,
											 ref cudnnPersistentRNNPlan_t plan);
                                             
			// Attaches the plan to the descriptor. 
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc,
										  cudnnPersistentRNNPlan_t plan);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan);



			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetRNNDescriptor_v6(cudnnHandle_t handle,
															cudnnRNNDescriptor_t rnnDesc,

												 int hiddenSize,

												 int numLayers,
												cudnnDropoutDescriptor_t dropoutDesc, // Between layers, not between recurrent steps.
                                                cudnnRNNInputMode_t inputMode,
												cudnnDirectionMode_t direction, 
                                                cudnnRNNMode_t mode,
												cudnnRNNAlgo_t algo, 
                                                cudnnDataType_t dataType);


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetRNNDescriptor(cudnnRNNDescriptor_t rnnDesc,
															int hiddenSize,
															int numLayers,
															cudnnDropoutDescriptor_t dropoutDesc, // Between layers, not between recurrent steps.
															cudnnRNNInputMode_t inputMode,
															cudnnDirectionMode_t direction,
															cudnnRNNMode_t mode,
															cudnnDataType_t dataType);



			// dataType in the RNN descriptor is used to determine math precision
			// dataType in weight descriptors and input descriptors is used to describe storage

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetRNNWorkspaceSize(cudnnHandle_t handle,
													 cudnnRNNDescriptor_t rnnDesc,
													 int seqLength,
													 ref cudnnTensorDescriptor_t xDesc,
													ref size_t sizeInBytes
													);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetRNNTrainingReserveSize(cudnnHandle_t handle,
														  cudnnRNNDescriptor_t rnnDesc,
														  int seqLength,
														  ref cudnnTensorDescriptor_t xDesc,
														  ref size_t sizeInBytes
													);

                                                    
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetRNNParamsSize(cudnnHandle_t handle,
												 cudnnRNNDescriptor_t rnnDesc,
												 cudnnTensorDescriptor_t xDesc,
												 ref size_t                     sizeInBytes,
                                                 cudnnDataType_t dataType
                                                    );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetRNNLinLayerMatrixParams(cudnnHandle_t handle,
							 cudnnRNNDescriptor_t rnnDesc,
							 int layer,
							 cudnnTensorDescriptor_t xDesc,
							 cudnnFilterDescriptor_t wDesc,
							 IntPtr w,
							 int linLayerID,
							 cudnnFilterDescriptor_t linLayerMatDesc, 
                             ref IntPtr linLayerMat
                             );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetRNNLinLayerBiasParams(cudnnHandle_t handle,
							 cudnnRNNDescriptor_t rnnDesc,
							 int layer,
							 cudnnTensorDescriptor_t xDesc,
							 cudnnFilterDescriptor_t wDesc,
							 IntPtr w,
							 int linLayerID,
							 cudnnFilterDescriptor_t linLayerBiasDesc, 
                             ref IntPtr linLayerBias                       
                             );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnRNNForwardInference(cudnnHandle_t handle,
													 cudnnRNNDescriptor_t rnnDesc,
													 int seqLength,
													 ref cudnnTensorDescriptor_t xDesc,
													 IntPtr x,
													 cudnnTensorDescriptor_t hxDesc,
													 IntPtr hx,
													 cudnnTensorDescriptor_t cxDesc,
													 IntPtr cx,
													 cudnnFilterDescriptor_t wDesc,
													 IntPtr w,
													 ref cudnnTensorDescriptor_t yDesc,
													IntPtr y,
													 cudnnTensorDescriptor_t hyDesc,
													IntPtr hy,
													 cudnnTensorDescriptor_t cyDesc,
													IntPtr cy,
													IntPtr workspace,
													size_t workSpaceSizeInBytes);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnRNNForwardTraining(cudnnHandle_t handle,
												   cudnnRNNDescriptor_t rnnDesc,
												   int seqLength,
												   ref cudnnTensorDescriptor_t xDesc,
												   IntPtr x,
												   cudnnTensorDescriptor_t hxDesc,
												   IntPtr hx,
												   cudnnTensorDescriptor_t cxDesc,
												   IntPtr cx,
												   cudnnFilterDescriptor_t wDesc,
												   IntPtr w,
												   ref cudnnTensorDescriptor_t yDesc,
												   IntPtr y,
												   cudnnTensorDescriptor_t hyDesc,
												   IntPtr hy,
												   cudnnTensorDescriptor_t cyDesc,
												   IntPtr cy,
												   IntPtr workspace,
												   size_t workSpaceSizeInBytes,
                                                   IntPtr reserveSpace,
												   size_t reserveSpaceSizeInBytes);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnRNNBackwardData(cudnnHandle_t handle,
												 cudnnRNNDescriptor_t rnnDesc,
												 int seqLength,
												 ref cudnnTensorDescriptor_t yDesc,
												 IntPtr y,
												 ref cudnnTensorDescriptor_t dyDesc,
												 IntPtr dy,
												 cudnnTensorDescriptor_t dhyDesc,
												 IntPtr dhy,
												 cudnnTensorDescriptor_t dcyDesc,
												 IntPtr dcy,
												 cudnnFilterDescriptor_t wDesc,
												 IntPtr w,
												 cudnnTensorDescriptor_t hxDesc,
												 IntPtr hx,
												 cudnnTensorDescriptor_t cxDesc,
												 IntPtr cx,
												 ref cudnnTensorDescriptor_t dxDesc,
												IntPtr dx,
												 cudnnTensorDescriptor_t dhxDesc,
												IntPtr dhx,
												 cudnnTensorDescriptor_t dcxDesc,
												IntPtr dcx,
												IntPtr workspace,
												size_t workSpaceSizeInBytes,
                                                IntPtr reserveSpace,
												size_t reserveSpaceSizeInBytes );


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnRNNBackwardWeights(cudnnHandle_t handle,
												   cudnnRNNDescriptor_t rnnDesc,
												   int seqLength,
												   ref cudnnTensorDescriptor_t xDesc,
												   IntPtr x,
												   cudnnTensorDescriptor_t hxDesc,
												   IntPtr hx,
												   ref cudnnTensorDescriptor_t yDesc,
												   IntPtr y,
												   IntPtr workspace,
												   size_t workSpaceSizeInBytes, 
                                                   cudnnFilterDescriptor_t dwDesc,
												   IntPtr dw,
												   IntPtr reserveSpace,
												   size_t reserveSpaceSizeInBytes );


			// DEPRECATED routines to be removed next release : 
			// User should use the non-suffixed version (which has the API and functionality of _v5 version)
			// Routines with _v4 suffix has the functionality of the non-suffixed routines in the CUDNN V5
			//
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetConvolution2dDescriptor_v4(
								cudnnConvolutionDescriptor_t convDesc,
								int pad_h,      // zero-padding height
								int pad_w,      // zero-padding width
								int u,          // vertical filter stride
								int v,          // horizontal filter stride
								int dilation_h, // filter dilation in the vertical dimension
								int dilation_w, // filter dilation in the horizontal dimension
								cudnnConvolutionMode_t mode);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnSetConvolution2dDescriptor_v5(cudnnConvolutionDescriptor_t convDesc,
																		 int pad_h,    // zero-padding height
																		 int pad_w,    // zero-padding width
																		 int u,   // vertical filter stride
																		 int v,   // horizontal filter stride
																		 int dilation_h, // filter dilation in the vertical dimension
																		 int dilation_w, // filter dilation in the horizontal dimension
																		 cudnnConvolutionMode_t mode,
																		 cudnnDataType_t computeType
																	   );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetConvolution2dDescriptor_v4(
								 cudnnConvolutionDescriptor_t convDesc,
								ref int pad_h,    // zero-padding height
								ref int pad_w,    // zero-padding width
								ref int u,        // vertical filter stride
								ref int v,        // horizontal filter stride
								ref int dilation_h, // filter dilation in the vertical dimension
								ref int dilation_w, // filter dilation in the horizontal dimension
								ref cudnnConvolutionMode_t mode);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudnnStatus_t cudnnGetConvolution2dDescriptor_v5(cudnnConvolutionDescriptor_t convDesc,
															ref int pad_h,    // zero-padding height
															ref int pad_w,    // zero-padding width
															ref int u,        // vertical filter stride
															ref int v,        // horizontal filter stride
															ref int dilation_h, // filter dilation in the vertical dimension
															ref int dilation_w, // filter dilation in the horizontal dimension
															ref cudnnConvolutionMode_t mode,
															ref cudnnDataType_t computeType
														 );
		}

		/// <summary>
		/// Maximum supported number of tensor dimensions
		/// </summary>
		public const int CUDNN_DIM_MAX = 8;

		/// <summary>
		/// minimum allowed lrnN
		/// </summary>
		public const int CUDNN_LRN_MIN_N = 1;

		/// <summary>
		/// maximum allowed lrnN
		/// </summary>
		public const int CUDNN_LRN_MAX_N = 16;

		/// <summary>
		/// minimum allowed lrnK
		/// </summary>
		public const double CUDNN_LRN_MIN_K = 1e-5;

		/// <summary>
		/// minimum allowed lrnBeta
		/// </summary>
		public const double CUDNN_LRN_MIN_BETA = 0.01;

		/// <summary>
		/// Minimum epsilon allowed to be used in the Batch Normalization formula
		/// </summary>
		public const double CUDNN_BN_MIN_EPSILON = 1e-5;

		// ----- C# Interface

		cudnnHandle_t handle = IntPtr.Zero;

		public cuDNN6() {
			handle = CreateHandle();
		}

		~cuDNN6() {
			Dispose();
		}

		public void Dispose() {
			if (handle != IntPtr.Zero) {
				DestroyHandle(handle);
				handle = IntPtr.Zero;
			}
		}

		public void SetStream(cudaStream_t stream) {
			CheckStatus(API.cudnnSetStream(handle, stream));
		}

		public cudaStream_t GetStream() {
			cudaStream_t stream = IntPtr.Zero;
			CheckStatus(API.cudnnGetStream(handle, ref stream));
			return stream;
		}

		public cudnnTensorDescriptor_t CreateTensorDescriptor() {
			cudnnTensorDescriptor_t tensorDesc = IntPtr.Zero;
			CheckStatus(API.cudnnCreateTensorDescriptor(ref tensorDesc));
			return tensorDesc;
		}

		public void SetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format, cudnnDataType_t dataType, int n, int c, int h, int w) {
			CheckStatus(API.cudnnSetTensor4dDescriptor(tensorDesc, format, dataType, n, c, h, w));
		}

		public void SetTensor4dDescriptorEx(cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int n, int c, int h, int w, int nStride, int cStride, int hStride, int wStride) {
			CheckStatus(API.cudnnSetTensor4dDescriptorEx(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride));
		}

		public void GetTensor4dDescriptor() {
			NotImplemented("GetTensor4dDescriptor");
		}

		public void SetTensorNdDescriptor() {
			NotImplemented("SetTensorNdDescriptor");
		}

		public void SetTensorNdDescriptorEx() {
			NotImplemented("SetTensorNdDescriptorEx");
		}

		public void GetTensorNdDescriptor() {
			NotImplemented("GetTensorNdDescriptor");
		}

		public size_t GetTensorSizeInBytes(cudnnTensorDescriptor_t tensorDesc) {
			size_t size = 0;
			CheckStatus(API.cudnnGetTensorSizeInBytes(tensorDesc, ref size));
			return size;
		}

		public void DestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
			CheckStatus(API.cudnnDestroyTensorDescriptor(tensorDesc));
		}

		public void TransformTensor(IntPtr alpha, cudnnTensorDescriptor_t xDesc, IntPtr x, IntPtr beta, cudnnTensorDescriptor_t yDesc, IntPtr y) {
			CheckStatus(API.cudnnTransformTensor(handle, alpha, xDesc, x, beta, yDesc, y));
		}

		public void AddTensor(IntPtr alpha, cudnnTensorDescriptor_t aDesc, IntPtr A, IntPtr beta, cudnnTensorDescriptor_t cDesc, IntPtr C) {
			CheckStatus(API.cudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C));
		}

		public cudnnOpTensorDescriptor_t CreateOpTensorDescriptor() {
			cudnnOpTensorDescriptor_t opTensorDesc = IntPtr.Zero;
			CheckStatus(API.cudnnCreateOpTensorDescriptor(ref opTensorDesc));
			return opTensorDesc;
		}

		public void SetOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t opTensorOp, cudnnDataType_t opTensorCompType, cudnnNanPropagation_t opTensorNanOpt) {
			CheckStatus(API.cudnnSetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt));
		}

		public void GetOpTensorDescriptor() {
			NotImplemented("GetOpTensorDescriptor");
		}

		public void DestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc) {
			CheckStatus(API.cudnnDestroyOpTensorDescriptor(opTensorDesc));
		}

		public void OpTensor(cudnnOpTensorDescriptor_t opTensorDesc, IntPtr alpha1, cudnnTensorDescriptor_t aDesc, IntPtr A, IntPtr alpha2, cudnnTensorDescriptor_t bDesc, IntPtr B, IntPtr beta, cudnnTensorDescriptor_t cDesc, IntPtr C) {
			CheckStatus(API.cudnnOpTensor(handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C));
		}

		public cudnnReduceTensorDescriptor_t CreateReduceTensorDescriptor() {
			cudnnReduceTensorDescriptor_t reduceTensorDesc = IntPtr.Zero;
			CheckStatus(API.cudnnCreateReduceTensorDescriptor(ref reduceTensorDesc));
			return reduceTensorDesc;
		}

		public void SetReduceTensorDescriptor() {
			NotImplemented("SetReduceTensorDescriptor");
		}

		public void GetReduceTensorDescriptor() {
			NotImplemented("GetReduceTensorDescriptor");
		}

		public void DestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc) {
			CheckStatus(API.cudnnDestroyReduceTensorDescriptor(reduceTensorDesc));
		}

		public size_t GetReductionIndicesSize(cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnTensorDescriptor_t aDesc, cudnnTensorDescriptor_t cDesc) {
			size_t sizeInBytes = 0;
			CheckStatus(API.cudnnGetReductionIndicesSize(handle, reduceTensorDesc, aDesc, cDesc, ref sizeInBytes));
			return sizeInBytes;
		}

		public size_t GetReductionWorkspaceSize(cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnTensorDescriptor_t aDesc, cudnnTensorDescriptor_t cDesc) {
			size_t sizeInBytes = 0;
			CheckStatus(API.cudnnGetReductionWorkspaceSize(handle, reduceTensorDesc, aDesc, cDesc, ref sizeInBytes));
			return sizeInBytes;
		}

		public void ReduceTensor(cudnnReduceTensorDescriptor_t reduceTensorDesc, IntPtr indices, size_t indicesSizeInBytes, IntPtr workspace, size_t workspaceSizeInBytes, IntPtr alpha, cudnnTensorDescriptor_t aDesc, IntPtr A, IntPtr beta, cudnnTensorDescriptor_t cDesc, IntPtr C) {
			CheckStatus(API.cudnnReduceTensor(handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace, workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C));
		}

		public void SetTensor(cudnnTensorDescriptor_t yDesc, IntPtr y, IntPtr valuePtr) {
			CheckStatus(API.cudnnSetTensor(handle, yDesc, y, valuePtr));
		}

		public void ScaleTensor(cudnnTensorDescriptor_t yDesc, IntPtr y, IntPtr alpha) {
			CheckStatus(API.cudnnSetTensor(handle, yDesc, y, alpha));
		}

		public cudnnFilterDescriptor_t CreateFilterDescriptor() {
			cudnnFilterDescriptor_t filterDesc = IntPtr.Zero;
			CheckStatus(API.cudnnCreateFilterDescriptor(ref filterDesc));
			return filterDesc;
		}

		public void SetFilter4dDescriptor() {
			NotImplemented("SetFilter4dDescriptor");
		}

		public void GetFilter4dDescriptor() {
			NotImplemented("GetFilter4dDescriptor");
		}

		public void SetFilterNdDescriptor() {
			NotImplemented("SetFilterNdDescriptor");
		}

		public void GetFilterNdDescriptor() {
			NotImplemented("GetFilterNdDescriptor");
		}

		public void DestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
			CheckStatus(API.cudnnDestroyFilterDescriptor(filterDesc));
		}

		public cudnnConvolutionDescriptor_t CreateConvolutionDescriptor() {
			cudnnConvolutionDescriptor_t convDesc = IntPtr.Zero;
			CheckStatus(API.cudnnCreateConvolutionDescriptor(ref convDesc));
			return convDesc;
		}

		public void SetConvolution2dDescriptor() {
			NotImplemented("SetConvolution2dDescriptor");
		}

		public void GetConvolution2dDescriptor() {
			NotImplemented("GetConvolution2dDescriptor");
		}

		public void GetConvolution2dForwardOutputDim() {
			NotImplemented("GetConvolution2dForwardOutputDim");
		}

		public void SetConvolutionNdDescriptor() {
			NotImplemented("SetConvolutionNdDescriptor");
		}

		public void GetConvolutionNdDescriptor() {
			NotImplemented("GetConvolutionNdDescriptor");
		}

		public void GetConvolutionNdForwardOutputDim() {
			NotImplemented("GetConvolutionNdForwardOutputDim");
		}

		public void DestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
			CheckStatus(API.cudnnDestroyConvolutionDescriptor(convDesc));
		}

		public void FindConvolutionForwardAlgorithm(cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t wDesc, cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t yDesc, int requestedAlgoCount) {
			int returnedAlgoCount = 0;
			cudnnConvolutionFwdAlgoPerf_t perfResults = new cudnnConvolutionFwdAlgoPerf_t();
			CheckStatus(API.cudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, ref returnedAlgoCount, ref perfResults));
		}

		public void FindConvolutionForwardAlgorithmEx() {
			NotImplemented("FindConvolutionForwardAlgorithmEx");
		}

		public void GetConvolutionForwardAlgorithm() {
			NotImplemented("GetConvolutionForwardAlgorithm");
		}

		public size_t GetConvolutionForwardWorkspaceSize(cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t wDesc, cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo) {
			size_t sizeInBytes = 0;
			CheckStatus(API.cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, algo, ref sizeInBytes));
			return sizeInBytes;
		}

		public void ConvolutionForward() {
			NotImplemented("ConvolutionForward");
		}

		public void ConvolutionBiasActivationForward() {
			NotImplemented("ConvolutionBiasActivationForward");
		}

		public void ConvolutionBackwardBias() {
			NotImplemented("ConvolutionBackwardBias");
		}

		public void FindConvolutionBackwardFilterAlgorithm() {
			NotImplemented("FindConvolutionBackwardFilterAlgorithm");
		}

		public void FindConvolutionBackwardFilterAlgorithmEx() {
			NotImplemented("FindConvolutionBackwardFilterAlgorithmEx");
		}

		public void GetConvolutionBackwardFilterAlgorithm() {
			NotImplemented("GetConvolutionBackwardFilterAlgorithm");
		}

		public size_t GetConvolutionBackwardFilterWorkspaceSize(cudnnTensorDescriptor_t xDesc, cudnnTensorDescriptor_t dyDesc, cudnnConvolutionDescriptor_t convDesc, cudnnFilterDescriptor_t gradDesc, cudnnConvolutionBwdFilterAlgo_t algo) {
			size_t sizeInBytes = 0;
			CheckStatus(API.cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, xDesc, dyDesc, convDesc, gradDesc, algo, ref sizeInBytes));
			return sizeInBytes;
		}

		public void ConvolutionBackwardFilter() {
			NotImplemented("ConvolutionBackwardFilter");
		}

		public void FindConvolutionBackwardDataAlgorithm() {
			NotImplemented("FindConvolutionBackwardDataAlgorithm");
		}

		public void FindConvolutionBackwardDataAlgorithmEx() {
			NotImplemented("FindConvolutionBackwardDataAlgorithmEx");
		}

		public void GetConvolutionBackwardDataAlgorithm() {
			NotImplemented("GetConvolutionBackwardDataAlgorithm");
		}

		public void GetConvolutionBackwardDataWorkspaceSize() {
			NotImplemented("GetConvolutionBackwardDataWorkspaceSize");
		}

		public void ConvolutionBackwardData() {
			NotImplemented("ConvolutionBackwardData");
		}

		public void Im2Col() {
			NotImplemented("Im2Col");
		}

		public void SoftmaxForward(cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode, IntPtr alpha, cudnnTensorDescriptor_t xDesc, IntPtr x, IntPtr beta, cudnnTensorDescriptor_t yDesc, IntPtr y) {
			CheckStatus(API.cudnnSoftmaxForward(handle, algo, mode, alpha, xDesc, x, beta, yDesc, y));
		}

		public void SoftmaxBackward(cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode, IntPtr alpha, cudnnTensorDescriptor_t yDesc, IntPtr y, cudnnTensorDescriptor_t dyDesc, IntPtr dy, IntPtr beta, cudnnTensorDescriptor_t dxDesc, IntPtr dx) {
			CheckStatus(API.cudnnSoftmaxBackward(handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx));
		}

		public cudnnPoolingDescriptor_t CreatePoolingDescriptor() {
			cudnnPoolingDescriptor_t poolingDesc = IntPtr.Zero;
			CheckStatus(API.cudnnCreatePoolingDescriptor(ref poolingDesc));
			return poolingDesc;
		}

		public void SetPooling2dDescriptor() {
			NotImplemented("SetPooling2dDescriptor");
		}

		public void GetPooling2dDescriptor() {
			NotImplemented("GetPooling2dDescriptor");
		}

		public void SetPoolingNdDescriptor() {
			NotImplemented("SetPoolingNdDescriptor");
		}

		public void GetPoolingNdDescriptor() {
			NotImplemented("GetPoolingNdDescriptor");
		}

		public void GetPoolingNdForwardOutputDim() {
			NotImplemented("GetPoolingNdForwardOutputDim");
		}

		public void GetPooling2dForwardOutputDim() {
			NotImplemented("GetPooling2dForwardOutputDim");
		}

		public void DestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc) {
			CheckStatus(API.cudnnDestroyPoolingDescriptor(poolingDesc));
		}

		public void PoolingForward() {
			NotImplemented("PoolingForward");
		}

		public void PoolingBackward() {
			NotImplemented("PoolingBackward");
		}

		public cudnnActivationDescriptor_t CreateActivationDescriptor() {
			cudnnActivationDescriptor_t activationDesc = IntPtr.Zero;
			CheckStatus(API.cudnnCreateActivationDescriptor(ref activationDesc));
			return activationDesc;
		}

		public void SetActivationDescriptor() {
			NotImplemented("SetActivationDescriptor");
		}

		public void GetActivationDescriptor() {
			NotImplemented("GetActivationDescriptor");
		}

		public void DestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc) {
			CheckStatus(API.cudnnDestroyActivationDescriptor(activationDesc));
		}

		public void ActivationForward() {
			NotImplemented("ActivationForward");
		}

		public void ActivationBackward() {
			NotImplemented("ActivationBackward");
		}

		public cudnnLRNDescriptor_t CreateLRNDescriptor() {
			cudnnLRNDescriptor_t lrnDesc = IntPtr.Zero;
			CheckStatus(API.cudnnCreateLRNDescriptor(ref lrnDesc));
			return lrnDesc;
		}

		public void SetLRNDescriptor() {
			NotImplemented("SetLRNDescriptor");
		}

		public void GetLRNDescriptor() {
			NotImplemented("GetLRNDescriptor");
		}

		public void DestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc) {
			CheckStatus(API.cudnnDestroyLRNDescriptor(lrnDesc));
		}

		public void LRNCrossChannelForward() {
			NotImplemented("LRNCrossChannelForward");
		}

		public void LRNCrossChannelBackward() {
			NotImplemented("LRNCrossChannelBackward");
		}

		public void DivisiveNormalizationForward() {
			NotImplemented("DivisiveNormalizationForward");
		}

		public void DivisiveNormalizationBackward() {
			NotImplemented("DivisiveNormalizationBackward");
		}

		public void DeriveBNTensorDescriptor() {
			NotImplemented("DeriveBNTensorDescriptor");
		}

		public void BatchNormalizationForwardTraining() {
			NotImplemented("BatchNormalizationForwardTraining");
		}

		public void BatchNormalizationForwardInference() {
			NotImplemented("BatchNormalizationForwardInference");
		}

		public void BatchNormalizationBackward() {
			NotImplemented("BatchNormalizationBackward");
		}

		public cudnnSpatialTransformerDescriptor_t CreateSpatialTransformerDescriptor() {
			cudnnSpatialTransformerDescriptor_t stDesc = IntPtr.Zero;
			CheckStatus(API.cudnnCreateSpatialTransformerDescriptor(ref stDesc));
			return stDesc;
		}

		public void SetSpatialTransformerNdDescriptor() {
			NotImplemented("SetSpatialTransformerNdDescriptor");
		}

		public void DestroySpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t stDesc) {
			CheckStatus(API.cudnnDestroySpatialTransformerDescriptor(stDesc));
		}

		public void SpatialTfGridGeneratorForward() {
			NotImplemented("SpatialTfGridGeneratorForward");
		}

		public void SpatialTfGridGeneratorBackward() {
			NotImplemented("SpatialTfGridGeneratorBackward");
		}

		public void SpatialTfSamplerForward() {
			NotImplemented("SpatialTfSamplerForward");
		}

		public void SpatialTfSamplerBackward() {
			NotImplemented("SpatialTfSamplerBackward");
		}

		public cudnnDropoutDescriptor_t CreateDropoutDescriptor() {
			cudnnDropoutDescriptor_t dropoutDesc = IntPtr.Zero;
			CheckStatus(API.cudnnCreateDropoutDescriptor(ref dropoutDesc));
			return dropoutDesc;
		}

		public void DestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc) {
			CheckStatus(API.cudnnDestroyDropoutDescriptor(dropoutDesc));
		}

		public size_t DropoutGetStatesSize() {
			size_t sizeInBytes = 0;
			CheckStatus(API.cudnnDropoutGetStatesSize(handle, ref sizeInBytes));
			return sizeInBytes;
		}

		public size_t DropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xdesc) {
			size_t sizeInBytes = 0;
			CheckStatus(API.cudnnDropoutGetReserveSpaceSize(xdesc, ref sizeInBytes));
			return sizeInBytes;
		}

		public void SetDropoutDescriptor() {
			NotImplemented("SetDropoutDescriptor");
		}

		public void DropoutForward() {
			NotImplemented("DropoutForward");
		}

		public void DropoutBackward() {
			NotImplemented("DropoutBackward");
		}

		public cudnnRNNDescriptor_t CreateRNNDescriptor() {
			cudnnRNNDescriptor_t rnnDesc = IntPtr.Zero;
			CheckStatus(API.cudnnCreateRNNDescriptor(ref rnnDesc));
			return rnnDesc;
		}

		public void DestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) {
			CheckStatus(API.cudnnDestroyRNNDescriptor(rnnDesc));
		}

		public cudnnPersistentRNNPlan_t CreatePersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc, int minibatch, cudnnDataType_t dataType) {
			cudnnPersistentRNNPlan_t plan = IntPtr.Zero;
			CheckStatus(API.cudnnCreatePersistentRNNPlan(rnnDesc, minibatch, dataType, ref plan));
			return plan;
		}

		public void SetPersistentRNNPlan() {
			NotImplemented("SetPersistentRNNPlan");
		}

		public void DestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan) {
			CheckStatus(API.cudnnDestroyPersistentRNNPlan(plan));
		}

		public void SetRNNDescriptor_v6() {
			NotImplemented("SetRNNDescriptor_v6");
		}

		public void SetRNNDescriptor() {
			NotImplemented("SetRNNDescriptor");
		}

		public void GetRNNWorkspaceSize() {
			NotImplemented("GetRNNWorkspaceSize");
		}

		public void GetRNNTrainingReserveSize() {
			NotImplemented("GetRNNTrainingReserveSize");
		}

		public void GetRNNParamsSize() {
			NotImplemented("GetRNNParamsSize");
		}

		public void GetRNNLinLayerMatrixParams() {
			NotImplemented("GetRNNLinLayerMatrixParams");
		}

		public void GetRNNLinLayerBiasParams() {
			NotImplemented("GetRNNLinLayerBiasParams");
		}

		public void RNNForwardInference() {
			NotImplemented("RNNForwardInference");
		}

		public void RNNForwardTraining() {
			NotImplemented("RNNForwardTraining");
		}

		public void RNNBackwardData() {
			NotImplemented("RNNBackwardData");
		}

		public void RNNBackwardWeights() {
			NotImplemented("RNNBackwardWeights");
		}

		public void SetConvolution2dDescriptor_v4() {
			NotImplemented("SetConvolution2dDescriptor_v4");
		}

		public void SetConvolution2dDescriptor_v5() {
			NotImplemented("SetConvolution2dDescriptor_v5");
		}

		public void GetConvolution2dDescriptor_v4() {
			NotImplemented("GetConvolution2dDescriptor_v4");
		}

		public void GetConvolution2dDescriptor_v5() {
			NotImplemented("GetConvolution2dDescriptor_v5");
		}


		public static string GetErrorString(cudnnStatus status) {
			IntPtr ptr = API.cudnnGetErrorString(status);
			return Marshal.PtrToStringAnsi(ptr);
		}

		public static int GetProperty(libraryPropertyType type) {
			int value = 0;
			CheckStatus(API.cudnnGetProperty(type, ref value));
			return value;
		}

		public static long GetVersion() {
			return API.cudnnGetVersion();
		}

		public static long GetCudartVersion() {
			return API.cudnnGetCudartVersion();
		}

		static void CheckStatus(cudnnStatus status) {
			if (status != cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new CudaException(status.ToString());
			}
		}

		cudnnHandle_t CreateHandle() {
			cudnnHandle_t handle = IntPtr.Zero;
			CheckStatus(API.cudnnCreate(ref handle));
			return handle;
		}

		void DestroyHandle(cudnnHandle_t handle) {
			CheckStatus(API.cudnnDestroy(handle));
		}

		void NotImplemented(string message) {
			throw new NotImplementedException(message);
		}
	}

	/// <summary>
	/// (cuDNN) CUDNN return codes.
	/// </summary>
	public enum cudnnStatus {
		CUDNN_STATUS_SUCCESS = 0,
		CUDNN_STATUS_NOT_INITIALIZED = 1,
		CUDNN_STATUS_ALLOC_FAILED = 2,
		CUDNN_STATUS_BAD_PARAM = 3,
		CUDNN_STATUS_INTERNAL_ERROR = 4,
		CUDNN_STATUS_INVALID_VALUE = 5,
		CUDNN_STATUS_ARCH_MISMATCH = 6,
		CUDNN_STATUS_MAPPING_ERROR = 7,
		CUDNN_STATUS_EXECUTION_FAILED = 8,
		CUDNN_STATUS_NOT_SUPPORTED = 9,
		CUDNN_STATUS_LICENSE_ERROR = 10,
		CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11
	}

	/// <summary>
	///(cuDNN) CUDNN data type.
	/// </summary>
	public enum cudnnDataType {
		CUDNN_DATA_FLOAT = 0,
		CUDNN_DATA_DOUBLE = 1,
		CUDNN_DATA_HALF = 2,
		CUDNN_DATA_INT8 = 3,
		CUDNN_DATA_INT32 = 4,
		CUDNN_DATA_INT8x4 = 5
	}

	/// <summary>
	/// (cuDNN) CUDNN propagate Nan.
	/// </summary>
	public enum cudnnNanPropagation {
		CUDNN_NOT_PROPAGATE_NAN = 0,
		CUDNN_PROPAGATE_NAN = 1,
	}

	/// <summary>
	/// (cuDNN) CUDNN Determinism.
	/// </summary>
	public enum cudnnDeterminism {
		CUDNN_NON_DETERMINISTIC = 0,
		CUDNN_DETERMINISTIC = 1,
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	public enum cudnnTensorFormat {
		/// <summary>
		/// row major (wStride = 1, hStride = w)
		/// </summary>
		CUDNN_TENSOR_NCHW = 0,

		/// <summary>
		/// feature maps interleaved ( cStride = 1 )
		/// </summary>
		CUDNN_TENSOR_NHWC = 1,

		/// <summary>
		/// each image point is vector of element of C : the length of the vector is carried by the data type
		/// </summary>
		CUDNN_TENSOR_NCHW_VECT_C = 2
	}

	/// <summary>
	/// (cuDNN) CUDNN OpTensor op type.
	/// </summary>
	public enum cudnnOpTensorOp {
		CUDNN_OP_TENSOR_ADD = 0,
		CUDNN_OP_TENSOR_MUL = 1,
		CUDNN_OP_TENSOR_MIN = 2,
		CUDNN_OP_TENSOR_MAX = 3,
		CUDNN_OP_TENSOR_SQRT = 4,
	}

	/// <summary>
	/// (cuDNN) CUDNN ReduceTensor op type.
	/// </summary>
	public enum cudnnReduceTensorOp {
		CUDNN_REDUCE_TENSOR_ADD = 0,
		CUDNN_REDUCE_TENSOR_MUL = 1,
		CUDNN_REDUCE_TENSOR_MIN = 2,
		CUDNN_REDUCE_TENSOR_MAX = 3,
		CUDNN_REDUCE_TENSOR_AMAX = 4,
		CUDNN_REDUCE_TENSOR_AVG = 5,
		CUDNN_REDUCE_TENSOR_NORM1 = 6,
		CUDNN_REDUCE_TENSOR_NORM2 = 7,
	}

	/// <summary>
	/// (cuDNN) CUDNN ReduceTensor indices type.
	/// </summary>
	public enum cudnnReduceTensorIndices {
		CUDNN_REDUCE_TENSOR_NO_INDICES = 0,
		CUDNN_REDUCE_TENSOR_FLATTENED_INDICES = 1,
	}

	/// <summary>
	/// (cuDNN) CUDNN tensor indices type size(all unsigned)
	/// Currently not supported, default is 32 bit unsigned.
	/// </summary>
	public enum cudnnIndicesType {
		CUDNN_32BIT_INDICES = 0,
		CUDNN_64BIT_INDICES = 1,
		CUDNN_16BIT_INDICES = 2,
		CUDNN_8BIT_INDICES = 3,
	}

	/// <summary>
	/// (cuDNN) convolution mode.
	/// </summary>
	public enum cudnnConvolutionMode {
		CUDNN_CONVOLUTION = 0,
		CUDNN_CROSS_CORRELATION = 1
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	public enum cudnnConvolutionFwdPreference {
		CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = 0,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1,
		CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2,
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	public enum cudnnConvolutionFwdAlgo {
		CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
		CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
		CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2,
		CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3,
		CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4,
		CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5,
		CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6,
		CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7,
		CUDNN_CONVOLUTION_FWD_ALGO_COUNT = 8,
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnConvolutionFwdAlgoPerf {
		public cudnnConvolutionFwdAlgo algo;
		public cudnnStatus_t status;
		public float time;
		public size_t memory;
		public cudnnDeterminism determinism;
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
		public int[] reserved;
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	public enum cudnnConvolutionBwdFilterPreference {
		CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0,
		CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = 1,
		CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2,
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	public enum cudnnConvolutionBwdFilterAlgo {
		/// <summary>
		/// non-deterministic
		/// </summary>
		CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0,
		CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1,
		CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2,
		/// <summary>
		/// non-deterministic, algo0 with workspace
		/// </summary>
		CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3,
		/// <summary>
		/// not implemented
		/// </summary>
		CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = 4,
		CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5,
		CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = 6,
		CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT = 7,
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnConvolutionBwdFilterAlgoPerf {
		public cudnnConvolutionBwdFilterAlgo algo;
		public cudnnStatus_t status;
		public float time;
		public size_t memory;
		public cudnnDeterminism determinism;
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
		public int[] reserved;
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	public enum cudnnConvolutionBwdDataPreference {
		CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = 0,
		CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = 1,
		CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2,
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	public enum cudnnConvolutionBwdDataAlgo {
		/// <summary>
		///  non-deterministic
		/// </summary>
		CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0,
		CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1,
		CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2,
		CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3,
		CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4,
		CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5,
		CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = 6,
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnConvolutionBwdDataAlgoPerf {
		public cudnnConvolutionBwdDataAlgo algo;
		public cudnnStatus_t status;
		public float time;
		public size_t memory;
		public cudnnDeterminism determinism;
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
		public int[] reserved;
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	public enum cudnnSoftmaxAlgorithm {
		/// <summary>
		/// straightforward implementation
		/// </summary>
		CUDNN_SOFTMAX_FAST = 0,
		/// <summary>
		/// subtract max from every point to avoid overflow
		/// </summary>
		CUDNN_SOFTMAX_ACCURATE = 1,
		CUDNN_SOFTMAX_LOG = 2
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	public enum cudnnSoftmaxMode {
		/// <summary>
		/// compute the softmax over all C, H, W for each N
		/// </summary>
		CUDNN_SOFTMAX_MODE_INSTANCE = 0,
		/// <summary>
		/// compute the softmax over all C for each H, W, N
		/// </summary>
		CUDNN_SOFTMAX_MODE_CHANNEL = 1
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	public enum cudnnPoolingMode {
		CUDNN_POOLING_MAX = 0,
		/// <summary>
		/// count for average includes padded values
		/// </summary>
		CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
		/// <summary>
		/// count for average does not include padded values
		/// </summary>
		CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2,
		CUDNN_POOLING_MAX_DETERMINISTIC = 3
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	public enum cudnnActivationMode {
		CUDNN_ACTIVATION_SIGMOID = 0,
		CUDNN_ACTIVATION_RELU = 1,
		CUDNN_ACTIVATION_TANH = 2,
		CUDNN_ACTIVATION_CLIPPED_RELU = 3,
		CUDNN_ACTIVATION_ELU = 4
	}

	/// <summary>
	/// (cuDNN) LRN layer mode
	/// </summary>
	public enum cudnnLRNMode {
		/// <summary>
		/// Normalize across tensor's dimA[1] dimension
		/// </summary>
		CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	public enum cudnnDivNormMode {
		CUDNN_DIVNORM_PRECOMPUTED_MEANS = 0
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	public enum cudnnBatchNormMode {
		/// <summary>
		/// bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per CHW...-slice, normalized over N slice)
		/// </summary>
		CUDNN_BATCHNORM_PER_ACTIVATION = 0,

		/// <summary>
		/// bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors)
		/// </summary>
		CUDNN_BATCHNORM_SPATIAL = 1
	}

	/// <summary>
	/// (cuDNN) APIs for spatial transformer network
	/// </summary>
	public enum cudnnSamplerType {
		CUDNN_SAMPLER_BILINEAR = 0
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	public enum cudnnRNNMode {
		/// <summary>
		/// Stock RNN with ReLu activation
		/// </summary>
		CUDNN_RNN_RELU = 0,
		/// <summary>
		/// Stock RNN with tanh activation
		/// </summary>
		CUDNN_RNN_TANH = 1,
		/// <summary>
		/// LSTM with no peephole connections
		/// </summary>
		CUDNN_LSTM = 2,
		/// <summary>
		/// Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1);
		/// </summary>
		CUDNN_GRU = 3
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	public enum cudnnDirectionMode {
		CUDNN_UNIDIRECTIONAL = 0,
		/// <summary>
		/// Using output concatination at each step. Do we also want to support output sum?
		/// </summary>
		CUDNN_BIDIRECTIONAL = 1
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	public enum cudnnRNNInputMode {
		CUDNN_LINEAR_INPUT = 0,
		CUDNN_SKIP_INPUT = 1
	}

	/// <summary>
	/// (cuDNN) 
	/// </summary>
	public enum cudnnRNNAlgo {
		CUDNN_RNN_ALGO_STANDARD = 0,
		CUDNN_RNN_ALGO_PERSIST_STATIC = 1,
		CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2
	}
}
