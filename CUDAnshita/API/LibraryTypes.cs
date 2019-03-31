namespace CUDAnshita {
	/// <summary>
	/// (library types) 
	/// </summary>
	public enum cudaDataType {
		///<summary>real as a half</summary>
		CUDA_R_16F = 2,
		///<summary>complex as a pair of half numbers</summary>
		CUDA_C_16F = 6,
		///<summary>real as a float</summary>
		CUDA_R_32F = 0,
		///<summary>complex as a pair of float numbers</summary>
		CUDA_C_32F = 4,
		///<summary>real as a double</summary>
		CUDA_R_64F = 1,
		///<summary>complex as a pair of double numbers</summary>
		CUDA_C_64F = 5,
		///<summary>real as a signed char</summary>
		CUDA_R_8I = 3,
		///<summary>complex as a pair of signed char numbers</summary>
		CUDA_C_8I = 7,
		///<summary>real as a unsigned char</summary>
		CUDA_R_8U = 8,
		///<summary>complex as a pair of unsigned char numbers</summary>
		CUDA_C_8U = 9,
		///<summary>real as a signed int</summary>
		CUDA_R_32I = 10,
		///<summary>complex as a pair of signed int numbers</summary>
		CUDA_C_32I = 11,
		///<summary>real as a unsigned int</summary>
		CUDA_R_32U = 12,
		///<summary>complex as a pair of unsigned int numbers</summary>
		CUDA_C_32U = 13
	}

	/// <summary>
	/// (library types) 
	/// </summary>
	public enum libraryPropertyType {
		///<summary>enumerant to query the major version</summary>
		MAJOR_VERSION,
		///<summary>enumerant to query the minor version</summary>
		MINOR_VERSION,
		///<summary>number to identify the patch level</summary>
		PATCH_LEVEL
	}
}
