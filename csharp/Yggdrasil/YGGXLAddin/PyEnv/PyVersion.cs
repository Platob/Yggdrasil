using System;
using System.Text.RegularExpressions;

namespace YGGXLAddin.PyEnv
{
    public readonly struct PyVersion
    {
        public readonly int Major;
        public readonly int Minor;
        public readonly int Patch;

        public PyVersion(int major, int minor, int patch)
        {
            Major = major;
            Minor = minor;
            Patch = patch;
        }

        public override string ToString() => $"{Major}.{Minor}.{Patch}";

        /// <summary>
        /// Parses the first X.Y.Z version found in the input string.
        /// Throws FormatException if parsing fails.
        /// </summary>
        public static PyVersion Parse(string text)
        {
            if (!TryParse(text, out var v))
                throw new FormatException("Could not parse version from: " + (text ?? "(null)"));
            return v;
        }

        /// <summary>
        /// Tries to parse the first X.Y.Z version found in the input string.
        /// Accepts strings like "3.12.1", "Python 3.11.7", "uv 0.4.30 (....)".
        /// </summary>
        public static bool TryParse(string text, out PyVersion version)
        {
            version = default;

            if (string.IsNullOrWhiteSpace(text))
                return false;

            // Find first X.Y.Z anywhere in the string.
            var m = Regex.Match(text, @"(\d+)\.(\d+)\.(\d+)");
            if (!m.Success)
                return false;

            if (!int.TryParse(m.Groups[1].Value, out var maj)) return false;
            if (!int.TryParse(m.Groups[2].Value, out var min)) return false;
            if (!int.TryParse(m.Groups[3].Value, out var pat)) return false;

            version = new PyVersion(maj, min, pat);
            return true;
        }
    }
}
