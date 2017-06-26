namespace CUDAnshita {
	public enum cudaDataType {
		CUDA_R_16F = 2,  /* real as a half */
		CUDA_C_16F = 6,  /* complex as a pair of half numbers */
		CUDA_R_32F = 0,  /* real as a float */
		CUDA_C_32F = 4,  /* complex as a pair of float numbers */
		CUDA_R_64F = 1,  /* real as a double */
		CUDA_C_64F = 5,  /* complex as a pair of double numbers */
		CUDA_R_8I = 3,  /* real as a signed char */
		CUDA_C_8I = 7,  /* complex as a pair of signed char numbers */
		CUDA_R_8U = 8,  /* real as a unsigned char */
		CUDA_C_8U = 9,  /* complex as a pair of unsigned char numbers */
		CUDA_R_32I = 10, /* real as a signed int */
		CUDA_C_32I = 11, /* complex as a pair of signed int numbers */
		CUDA_R_32U = 12, /* real as a unsigned int */
		CUDA_C_32U = 13  /* complex as a pair of unsigned int numbers */
	}

	public enum libraryPropertyType {
		MAJOR_VERSION,
		MINOR_VERSION,
		PATCH_LEVEL
	}
}
