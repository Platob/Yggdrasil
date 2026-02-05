using System.IO;
using NUnit.Framework;
using YGGXLAddin.Python;

namespace YGGXLAddin.PyEnv.Tests
{
    [TestFixture]
    public class PyEnvManagerTests
    {
        [Test]
        public void SystemDefaultTest()
        {
            var dft = PyEnvManager.SystemDefault();

            Assert.That(dft, Is.Not.Null);
            Assert.That(dft.Metadata, Is.Not.Null);

            Assert.That(dft.ExePath, Is.Not.Null.And.Not.Empty);
            Assert.That(dft.Name, Is.Not.Null.And.Not.Empty);

            Assert.That(File.Exists(dft.ExePath), Is.True);

            Assert.That(dft.Version.Major, Is.GreaterThanOrEqualTo(2));
            Assert.That(dft.Version.Minor, Is.GreaterThanOrEqualTo(0));
            Assert.That(dft.Version.Patch, Is.GreaterThanOrEqualTo(0));
        }

        [Test]
        public void SystemDefault_IsSingleton()
        {
            var a = PyEnvManager.SystemDefault();
            var b = PyEnvManager.SystemDefault();

            Assert.That(ReferenceEquals(a, b), Is.True);
        }

        [Test]
        public void SystemDefault_RunCode_Works()
        {
            var env = PyEnvManager.SystemDefault();

            var res = env.RunCode("import sys; print(sys.executable); print(sys.version_info[0])");

            Assert.That(res.ExitCode, Is.EqualTo(0));
            Assert.That(res.StdOut, Is.Not.Null.And.Not.Empty);
        }

        [Test]
        public void Run_InvalidArgs_ReturnsNonZero()
        {
            var env = PyEnvManager.SystemDefault();
            var res = env.RunCode("import sys; sys.exit(42)");

            Assert.That(res.ExitCode, Is.EqualTo(42));
        }

        [Test]
        public void FindUVPath_InstallsOrFindsUv()
        {
            var env = PyEnvManager.SystemDefault();

            // Mutates system python. Only enable locally.
            // Assert.Inconclusive("Disabled by default: installs uv into system python.");

            var uv = env.FindUVPath(true);
            Assert.That(uv, Is.Not.Null.And.Not.Empty);
            Assert.That(File.Exists(uv), Is.True);
        }
    }
}
