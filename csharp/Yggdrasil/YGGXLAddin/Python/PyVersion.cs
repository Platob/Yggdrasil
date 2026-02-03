using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace YGGXLAddin.Python
{
    public readonly struct PyVersion
    {
        public readonly int Major;
        public readonly int Minor;
        public readonly int Patch;

        // Default patch per (Major, Minor) when input is only "X.Y"
        // Example: "3.12" -> Patch = 12 (per your example)
        private static readonly IReadOnlyDictionary<(int Major, int Minor), int> DefaultPatchByMajorMinor
            = new Dictionary<(int, int), int>
            {
                [(3, 14)] = 2,
                [(3, 13)] = 11,
                [(3, 12)] = 12,
                [(3, 11)] = 14,
                [(3, 10)] = 19,
                // add whatever you actually want here
            };

        public PyVersion(int major, int minor, int patch)
        {
            Major = major;
            Minor = minor;
            Patch = patch;
        }

        public override string ToString() => $"{Major}.{Minor}.{Patch}";

        public static PyVersion Parse(string text)
        {
            if (!TryParse(text, out var v))
                throw new FormatException("Could not parse version from: " + (text ?? "(null)"));
            return v;
        }

        /// <summary>
        /// Tries to parse the first X.Y or X.Y.Z version found in the input string.
        /// If patch is missing, defaults patch via DefaultPatchByMajorMinor.
        /// </summary>
        public static bool TryParse(string text, out PyVersion version)
        {
            version = default;

            if (string.IsNullOrWhiteSpace(text))
                return false;

            // Find first X.Y[.Z] anywhere in the string.
            // Groups: 1=major, 2=minor, 3=patch (optional)
            var m = Regex.Match(text, @"(\d+)\.(\d+)(?:\.(\d+))?");
            if (!m.Success)
                return false;

            if (!int.TryParse(m.Groups[1].Value, out var maj)) return false;
            if (!int.TryParse(m.Groups[2].Value, out var min)) return false;

            int pat;
            if (m.Groups[3].Success)
            {
                if (!int.TryParse(m.Groups[3].Value, out pat)) return false;
            }
            else
            {
                // Patch missing -> default from dict
                if (!DefaultPatchByMajorMinor.TryGetValue((maj, min), out pat))
                    pat = 0; // fallback policy if no mapping exists
            }

            version = new PyVersion(maj, min, pat);
            return true;
        }
    }
}
