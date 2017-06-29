using System;
using System.Runtime.InteropServices;

namespace CUDAnshita {
	internal class MarshalUtil {
		public static void Copy<T>(IntPtr source, T[] destination, int startIndex, int length) {
			Type type = typeof(T);

			if (type == typeof(byte)) {
				Marshal.Copy(source, destination as byte[], startIndex, length);
			} else if (type == typeof(char)) {
				Marshal.Copy(source, destination as char[], startIndex, length);
			} else if (type == typeof(double)) {
				Marshal.Copy(source, destination as double[], startIndex, length);
			} else if (type == typeof(float)) {
				Marshal.Copy(source, destination as float[], startIndex, length);
			} else if (type == typeof(int)) {
				Marshal.Copy(source, destination as int[], startIndex, length);
			} else if (type == typeof(IntPtr)) {
				Marshal.Copy(source, destination as IntPtr[], startIndex, length);
			} else if (type == typeof(long)) {
				Marshal.Copy(source, destination as long[], startIndex, length);
			} else if (type == typeof(short)) {
				Marshal.Copy(source, destination as short[], startIndex, length);
			}
		}

		public static void Copy<T>(T[] source, int startIndex, IntPtr destination, int length) {
			Type type = typeof(T);

			if (type == typeof(byte)) {
				Marshal.Copy(source as byte[], startIndex, destination, length);
			} else if (type == typeof(char)) {
				Marshal.Copy(source as char[], startIndex, destination, length);
			} else if (type == typeof(double)) {
				Marshal.Copy(source as double[], startIndex, destination, length);
			} else if (type == typeof(float)) {
				Marshal.Copy(source as float[], startIndex, destination, length);
			} else if (type == typeof(int)) {
				Marshal.Copy(source as int[], startIndex, destination, length);
			} else if (type == typeof(IntPtr)) {
				Marshal.Copy(source as IntPtr[], startIndex, destination, length);
			} else if (type == typeof(long)) {
				Marshal.Copy(source as long[], startIndex, destination, length);
			} else if (type == typeof(short)) {
				Marshal.Copy(source as short[], startIndex, destination, length);
			}
		}
	}
}
