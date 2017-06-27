using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace CUDAnshita_Sample {
	public class Matrix2D : IEnumerable<List<double>> {
#if NET20
		public delegate void Action();
		public delegate void Action<T1, T2>(T1 arg1, T2 arg2);
		public delegate void Action<T1, T2, T3>(T1 arg1, T2 arg2, T3 arg3);
		public delegate void Action<T1, T2, T3, T4>(T1 arg1, T2 arg2, T3 arg3, T4 arg4);
		public delegate TResult Func<TResult>();
		public delegate TResult Func<T, TResult>(T arg);
		public delegate TResult Func<T1, T2, TResult>(T1 arg1, T2 arg2);
		public delegate TResult Func<T1, T2, T3, TResult>(T1 arg1, T2 arg2, T3 arg3);
		public delegate TResult Func<T1, T2, T3, T4, TResult>(T1 arg1, T2 arg2, T3 arg3, T4 arg4);
#endif
		protected double[] _Data;
		protected int _Width;
		protected int _Height;

		public double this[int i] {
			get { return _Data[i]; }
			set { _Data[i] = value; }
		}
		public double this[int x, int y] {
			get { return _Data[x + y * _Width]; }
			set { _Data[x + y * _Width] = value; }
		}

		public int Width {
			get { return _Width; }
		}
		public int Height {
			get { return _Height; }
		}
		public int Count {
			get { return _Data.Length; }
		}
		public double[] Data {
			get { return _Data; }
		}


		public static Matrix2D operator -(Matrix2D value) {
			var result = new Matrix2D(value.Width, value.Height);
			result.ForEach((matrix, x, y) => {
				result[x, y] = -value[x, y];
				return false;
			});
			return result;
		}

		public static Matrix2D operator +(Matrix2D value1, Matrix2D value2) {
			if (value1 == null || value2 == null) {
				return null;
			}
			var width = Math.Max(value1.Width, value2.Width);
			var height = Math.Max(value1.Height, value2.Height);
			var result = new Matrix2D(width, height);
			result.ForEach((matrix, x, y) => {
				result[x, y] = value1[x, y] + value2[x, y];
			});
			return result;
		}

		public static Matrix2D operator +(Matrix2D value1, double value2) {
			var result = new Matrix2D(value1.Width, value1.Height);
			result.ForEach((matrix, x, y) => {
				result[x, y] = value1[x, y] + value2;
				return false;
			});
			return result;
		}

		public static Matrix2D operator +(double value1, Matrix2D value2) {
			var result = new Matrix2D(value2.Width, value2.Height);
			result.ForEach((matrix, x, y) => {
				result[x, y] = value1 + value2[x, y];
				return false;
			});
			return result;
		}

		public static Matrix2D operator -(Matrix2D value1, Matrix2D value2) {
			if (value1 == null || value2 == null) {
				return null;
			}
			var width = Math.Max(value1.Width, value2.Width);
			var height = Math.Max(value1.Height, value2.Height);
			var result = new Matrix2D(width, height);
			result.ForEach((matrix, x, y) => {
				result[x, y] = value1[x, y] - value2[x, y];
				return false;
			});
			return result;
		}

		public static Matrix2D operator -(Matrix2D value1, double value2) {
			var result = new Matrix2D(value1.Width, value1.Height);
			result.ForEach((matrix, x, y) => {
				result[x, y] = value1[x, y] - value2;
				return false;
			});
			return result;
		}

		public static Matrix2D operator -(double value1, Matrix2D value2) {
			var result = new Matrix2D(value2.Width, value2.Height);
			result.ForEach((matrix, x, y) => {
				result[x, y] = value1 - value2[x, y];
				return false;
			});
			return result;
		}

		public static Matrix2D operator *(Matrix2D value1, Matrix2D value2) {
			var width = Math.Max(value1.Width, value2.Width);
			var height = Math.Max(value1.Height, value2.Height);
			var result = new Matrix2D(width, height);
			result.ForEach((matrix, x, y) => {
				result[x, y] = value1[x, y] * value2[x, y];
				return false;
			});
			return result;
		}

		public static Matrix2D operator *(Matrix2D value1, double value2) {
			var result = new Matrix2D(value1.Width, value1.Height);
			result.ForEach((matrix, x, y) => {
				result[x, y] = value1[x, y] * value2;
				return false;
			});
			return result;
		}

		public static Matrix2D operator *(double value1, Matrix2D value2) {
			var result = new Matrix2D(value2.Width, value2.Height);
			result.ForEach((matrix, x, y) => {
				result[x, y] = value1 * value2[x, y];
				return false;
			});
			return result;
		}

		public static Matrix2D operator /(Matrix2D value1, double value2) {
			var result = new Matrix2D(value1.Width, value1.Height);
			result.ForEach((matrix, x, y) => {
				result[x, y] = value1[x, y] / value2;
				return false;
			});
			return result;
		}

		public static Matrix2D operator /(double value1, Matrix2D value2) {
			var result = new Matrix2D(value2.Width, value2.Height);
			result.ForEach((matrix, x, y) => {
				result[x, y] = value1 / value2[x, y];
				return false;
			});
			return result;
		}

		public static Matrix2D operator <(Matrix2D value1, double value2) {
			if (value1 == null) {
				return null;
			}
			var result = new Matrix2D(value1.Width, value1.Height);
			result.ForEach((matrix, x, y) => {
				result[x, y] = value1[x, y] < value2 ? 1: 0;
				return false;
			});
			return result;
		}

		public static Matrix2D operator >(Matrix2D value1, double value2) {
			if (value1 == null) {
				return null;
			}
			var result = new Matrix2D(value1.Width, value1.Height);
			result.ForEach((matrix, x, y) => {
				result[x, y] = value1[x, y] > value2 ? 1 : 0;
				return false;
			});
			return result;
		}

		public static Matrix2D operator <=(Matrix2D value1, double value2) {
			if (value1 == null) {
				return null;
			}
			var result = new Matrix2D(value1.Width, value1.Height);
			result.ForEach((matrix, x, y) => {
				result[x, y] = value1[x, y] <= value2 ? 1 : 0;
				return false;
			});
			return result;
		}

		public static Matrix2D operator >=(Matrix2D value1, double value2) {
			if (value1 == null) {
				return null;
			}
			var result = new Matrix2D(value1.Width, value1.Height);
			result.ForEach((matrix, x, y) => {
				result[x, y] = value1[x, y] >= value2 ? 1 : 0;
				return false;
			});
			return result;
		}

		public Matrix2D Equals(Matrix2D value) {
			if (value == null) {
				return null;
			}
			var result = new Matrix2D(Width, Height);
			result.ForEach((matrix, x, y) => {
				result[x, y] = this[x, y] == value[x, y] ? 1 : 0;
				return false;
			});
			return result;
		}

		public Matrix2D(int width, int height) {
			width = Math.Max(1, width);
			height = Math.Max(1, height);
			_Width = width;
			_Height = height;
			_Data = new double[width * height];
		}

		public Matrix2D(int width) : this(width, 1) {
		}

		public Matrix2D(List<double> list) {
			if (list == null) {
				throw new ArgumentNullException();
			}
			_Width = list.Count;
			_Height = 1;
			_Data = new double[_Width];
			for (int i = 0; i < _Width; i++) {
				_Data[i] = list[i];
			}
		}

		public Matrix2D(double[] list) {
			if (list == null) {
				throw new ArgumentNullException();
			}
			_Width = list.Length;
			_Height = 1;
			_Data = new double[_Width];
			for (int i = 0; i < _Width; i++) {
				_Data[i] = list[i];
			}
		}

		public Matrix2D Clone() {
			var result = new Matrix2D(_Width, _Height);
			_Data.CopyTo(result._Data, 0);
			return result;
		}

		public static double Sum(Matrix2D matrix) {
			double result = 0;
			matrix.ForEach((m, x, y) => {
				result = result + matrix[x, y];
				return false;
			});
			return result;
		}

		public void Set(params double[] values) {
			var length = values.Length;
			for (var i = 0; i < length; i++) {
				if (i >= _Data.Length) {
					break;
				}
				_Data[i] = values[i];
			}
		}

		public List<int> Shape() {
			var result = new List<int>();
			result.Add(Width);
			result.Add(Height);
			return result;
		}

		public List<double> Flatten() {
			var result = new List<double>();
			ForEach((matrix, x, y) => {
				result.Add(matrix[x, y]);
				return false;
			});
			return result;
		}

		public double Sum() {
			double result = 0;
			ForEach((m, x, y) => {
				result += this[x, y];
				return false;
			});
			return result;
		}

		public double Max() {
			double result = 0;
			if (Count <= 0) {
				return result;
			}
			result = this[0, 0];
			ForEach((m, x, y) => {
				var value = this[x, y];
				if (value > result) {
					result = value;
				}
				return false;
			});
			return result;
		}

		public int ArgMax() {
			var result = -1;
			if (Count <= 0) {
				return result;
			}
			var index = 0;
			var max = this[0, 0];
			result = 0;
			ForEach((m, x, y) => {
				var value = this[x, y];
				if (value > max) {
					max = value;
					result = index;
				}
				index++;
				return false;
			});
			return result;
		}

		public Matrix2D Dot(Matrix2D value) {
			if (Width != value.Height) {
				throw new ArgumentException(string.Format(
					"ValueError: shapes ({0}, {1}) and ({2}, {3}) not aligned",
					Width, Height,
					value.Width, value.Height
				));
			}
			var result = new Matrix2D(value.Width, Height);
			result.ForEach((matrix, x, y) => {
				double total = 0;
				int count = Width;
				for (int i = 0; i < count; i++) {
					var r = this[i, y] * value[x, i];
					total += r;
				}
				result[x, y] = total;
			});
			return result;
		}

		public Matrix2D RandomChoice(int count) {
			if (count <= 0) {
				return null;
			}
			count = Math.Min(count, Height);
			var result = new Matrix2D(Width, count);
			var index = new List<int>();
			for (var i = 0; i < Height; i++) {
				index.Add(i);
			}
			var random = new Random();
			for (var i = 0; i < Height; i++) {
				var n = random.Next(Height);
				var tmp = index[n];
				index[n] = index[i];
				index[i] = tmp;
			}
			for (var i = 0; i < count; i++) {
				var n = index[i];
				for (var x = 0; x < Width; x++) {
					result[x, i] = this[x, n];
				}
			}
			return result;
		}

		public void ForEach(Func<Matrix2D, int, int, bool> action) {
			int width = _Width;
			int height = _Height;
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					if (action(this, x, y)) {
						return;
					}
				}
			}
		}

		public void ForEach(Action<Matrix2D, int, int> action) {
			int width = _Width;
			int height = _Height;
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					action(this, x, y);
				}
			}
		}

		public Matrix2D Every(Func<double, double> predicate) {
			int width = _Width;
			int height = _Height;
			var result = new Matrix2D(width, height);
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					result[x, y] = predicate(this[x, y]);
				}
			}
			return result;
		}

		public Matrix2D Every(double left, Func<double, double, double> predicate) {
			int width = _Width;
			int height = _Height;
			var result = new Matrix2D(width, height);
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					result[x, y] = predicate(left, this[x, y]);
				}
			}
			return result;
		}

		public Matrix2D Every(Func<double, double, double> predicate, double right) {
			int width = _Width;
			int height = _Height;
			var result = new Matrix2D(width, height);
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					result[x, y] = predicate(this[x, y], right);
				}
			}
			return result;
		}

		public override string ToString() {
			var text = new StringBuilder();
			text.Append("[");

			int width = _Width;
			int height = _Height;
			for (var y = 0; y < height; y++) {
				if (height > 1) {
					text.AppendLine();
					text.Append("\t[");
				}
				for (var x = 0; x < width; x++) {
					text.Append(this[x, y].ToString());
					if (x + 1 != _Width) {
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

		public List<double> Row(int y) {
			if (y >= _Height) {
				return null;
			}
			var data = new double[_Width];
			_Data.CopyTo(data, 0);
			return new List<double>(data);
		}

		public IEnumerator GetEnumerator() {
			int height = _Height;
			for (int y = 0; y < height; y++) {
				yield return Row(y);
			}
		}

		IEnumerator<List<double>> IEnumerable<List<double>>.GetEnumerator() {
			return (IEnumerator<List<double>>)GetEnumerator();
		}
	}
}
