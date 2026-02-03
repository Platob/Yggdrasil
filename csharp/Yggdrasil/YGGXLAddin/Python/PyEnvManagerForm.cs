using System;
using System.Drawing;
using System.Windows.Forms;
using YGGXLAddin.Python;

namespace YGGXLAddin
{
    public sealed class PyEnvManagerForm : Form
    {
        public readonly PyEnvManager _manager;
        private readonly ListView _envList;
        private readonly ListView _packageList;
        private readonly TextBox _nameInput;
        private readonly TextBox _versionInput;
        private readonly TextBox _packageInput;
        private readonly Label _baseDirLabel;
        private readonly Button _refreshButton;
        private readonly Button _createButton;
        private readonly Button _deleteButton;
        private readonly Button _refreshPackagesButton;
        private readonly Button _installPackageButton;
        private readonly Button _upgradePackageButton;
        private readonly Button _uninstallPackageButton;

        public PyEnvManagerForm()
        {
            _manager = PyEnvManager.Instance;

            Text = "Python Environments";
            MinimumSize = new Size(720, 420);
            StartPosition = FormStartPosition.CenterParent;

            var layout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 6,
                Padding = new Padding(12)
            };

            layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 45));
            layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 55));
            layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));

            _baseDirLabel = new Label
            {
                AutoSize = true,
                Text = $"Base directory: {_manager.BaseDir}"
            };

            _envList = new ListView
            {
                Dock = DockStyle.Fill,
                View = View.Details,
                FullRowSelect = true,
                MultiSelect = false
            };
            _envList.Columns.Add("Name", 140);
            _envList.Columns.Add("Version", 100);
            _envList.Columns.Add("Executable", 380);
            _envList.SelectedIndexChanged += (_, __) => RefreshPackages();

            var createPanel = new TableLayoutPanel
            {
                Dock = DockStyle.Top,
                ColumnCount = 5,
                AutoSize = true
            };
            createPanel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            createPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 40));
            createPanel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            createPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 30));
            createPanel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));

            createPanel.Controls.Add(new Label
            {
                AutoSize = true,
                Text = "Name:",
                TextAlign = ContentAlignment.MiddleLeft
            }, 0, 0);

            _nameInput = new TextBox { Dock = DockStyle.Fill };
            createPanel.Controls.Add(_nameInput, 1, 0);

            createPanel.Controls.Add(new Label
            {
                AutoSize = true,
                Text = "Python:",
                TextAlign = ContentAlignment.MiddleLeft
            }, 2, 0);

            _versionInput = new TextBox { Dock = DockStyle.Fill, Text = "3.12.12" };
            createPanel.Controls.Add(_versionInput, 3, 0);

            _createButton = new Button
            {
                Text = "Create",
                AutoSize = true
            };
            _createButton.Click += (_, __) => CreateEnvironment();
            createPanel.Controls.Add(_createButton, 4, 0);

            _packageList = new ListView
            {
                Dock = DockStyle.Fill,
                View = View.Details,
                FullRowSelect = true,
                MultiSelect = false
            };
            _packageList.Columns.Add("Package", 220);
            _packageList.Columns.Add("Version", 120);

            var packageActionPanel = new TableLayoutPanel
            {
                Dock = DockStyle.Top,
                ColumnCount = 6,
                AutoSize = true
            };
            packageActionPanel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            packageActionPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 40));
            packageActionPanel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            packageActionPanel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            packageActionPanel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            packageActionPanel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));

            packageActionPanel.Controls.Add(new Label
            {
                AutoSize = true,
                Text = "Package:",
                TextAlign = ContentAlignment.MiddleLeft
            }, 0, 0);

            _packageInput = new TextBox { Dock = DockStyle.Fill };
            packageActionPanel.Controls.Add(_packageInput, 1, 0);

            _installPackageButton = new Button { Text = "Install", AutoSize = true };
            _installPackageButton.Click += (_, __) => InstallPackage();
            packageActionPanel.Controls.Add(_installPackageButton, 2, 0);

            _upgradePackageButton = new Button { Text = "Upgrade", AutoSize = true };
            _upgradePackageButton.Click += (_, __) => UpgradePackage();
            packageActionPanel.Controls.Add(_upgradePackageButton, 3, 0);

            _uninstallPackageButton = new Button { Text = "Uninstall", AutoSize = true };
            _uninstallPackageButton.Click += (_, __) => UninstallPackage();
            packageActionPanel.Controls.Add(_uninstallPackageButton, 4, 0);

            _refreshPackagesButton = new Button { Text = "Refresh Packages", AutoSize = true };
            _refreshPackagesButton.Click += (_, __) => RefreshPackages();
            packageActionPanel.Controls.Add(_refreshPackagesButton, 5, 0);

            var actionPanel = new FlowLayoutPanel
            {
                Dock = DockStyle.Top,
                AutoSize = true,
                FlowDirection = FlowDirection.LeftToRight
            };

            _refreshButton = new Button { Text = "Refresh", AutoSize = true };
            _refreshButton.Click += (_, __) => RefreshEnvs();
            actionPanel.Controls.Add(_refreshButton);

            _deleteButton = new Button { Text = "Delete Selected", AutoSize = true };
            _deleteButton.Click += (_, __) => DeleteSelected();
            actionPanel.Controls.Add(_deleteButton);

            layout.Controls.Add(_baseDirLabel, 0, 0);
            layout.Controls.Add(_envList, 0, 1);
            layout.Controls.Add(createPanel, 0, 2);
            layout.Controls.Add(_packageList, 0, 3);
            layout.Controls.Add(packageActionPanel, 0, 4);
            layout.Controls.Add(actionPanel, 0, 5);

            Controls.Add(layout);

            RefreshEnvs();
        }

        private void RefreshEnvs()
        {
            try
            {
                _manager.Reload();
                _envList.Items.Clear();

                foreach (var pair in _manager.Envs)
                {
                    var env = pair.Value;
                    var item = new ListViewItem(env.Name);
                    item.SubItems.Add(env.Version.ToString());
                    item.SubItems.Add(env.ExePath);
                    _envList.Items.Add(item);
                }

                RefreshPackages();
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, ex.Message, "Failed to refresh", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void CreateEnvironment()
        {
            var name = (_nameInput.Text ?? "").Trim();
            var versionText = (_versionInput.Text ?? "").Trim();

            if (string.IsNullOrWhiteSpace(name))
            {
                MessageBox.Show(this, "Environment name is required.", "Missing name", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            try
            {
                _manager.Create(name, versionText);
                RefreshEnvs();
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, ex.Message, "Create failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void DeleteSelected()
        {
            if (_envList.SelectedItems.Count == 0)
            {
                MessageBox.Show(this, "Select an environment to delete.", "No selection", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            var name = _envList.SelectedItems[0].Text;
            var confirm = MessageBox.Show(
                this,
                $"Delete environment '{name}'? This removes the folder from disk.",
                "Confirm delete",
                MessageBoxButtons.YesNo,
                MessageBoxIcon.Warning);

            if (confirm != DialogResult.Yes)
                return;

            try
            {
                _manager.Delete(name);
                RefreshEnvs();
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, ex.Message, "Delete failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private PyEnv GetSelectedEnv()
        {
            if (_envList.SelectedItems.Count == 0)
                return null;

            var name = _envList.SelectedItems[0].Text;
            return _manager.TryGet(name, out var env) ? env : null;
        }

        private void RefreshPackages()
        {
            _packageList.Items.Clear();
            var env = GetSelectedEnv();
            if (env == null)
                return;

            try
            {
                var res = env.Run("-m pip list --format=freeze");
                if (res.ExitCode != 0)
                {
                    MessageBox.Show(this, res.StdErr, "Package list failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }

                var lines = (res.StdOut ?? "").Split(new[] { "\r\n", "\n" }, StringSplitOptions.RemoveEmptyEntries);
                foreach (var line in lines)
                {
                    var parts = line.Split(new[] { "==" }, StringSplitOptions.None);
                    var item = new ListViewItem(parts[0]);
                    item.SubItems.Add(parts.Length > 1 ? parts[1] : "");
                    _packageList.Items.Add(item);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, ex.Message, "Package list failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void InstallPackage()
        {
            var env = GetSelectedEnv();
            if (env == null)
            {
                MessageBox.Show(this, "Select an environment first.", "No environment", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            var package = (_packageInput.Text ?? "").Trim();
            if (string.IsNullOrWhiteSpace(package))
            {
                MessageBox.Show(this, "Enter a package spec to install.", "Missing package", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            try
            {
                var res = env.PipInstall(packageSpec: package);
                if (res.ExitCode != 0)
                {
                    MessageBox.Show(this, res.StdErr, "Install failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }

                RefreshPackages();
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, ex.Message, "Install failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void UpgradePackage()
        {
            var env = GetSelectedEnv();
            if (env == null)
            {
                MessageBox.Show(this, "Select an environment first.", "No environment", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            var package = (_packageInput.Text ?? "").Trim();
            if (string.IsNullOrWhiteSpace(package))
            {
                MessageBox.Show(this, "Enter a package spec to upgrade.", "Missing package", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            try
            {
                var res = env.PipInstall(packageSpec: package, upgrade: true);
                if (res.ExitCode != 0)
                {
                    MessageBox.Show(this, res.StdErr, "Upgrade failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }

                RefreshPackages();
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, ex.Message, "Upgrade failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void UninstallPackage()
        {
            var env = GetSelectedEnv();
            if (env == null)
            {
                MessageBox.Show(this, "Select an environment first.", "No environment", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            var package = (_packageInput.Text ?? "").Trim();
            if (string.IsNullOrWhiteSpace(package))
            {
                if (_packageList.SelectedItems.Count > 0)
                    package = _packageList.SelectedItems[0].Text;
            }

            if (string.IsNullOrWhiteSpace(package))
            {
                MessageBox.Show(this, "Select or enter a package name to uninstall.", "Missing package", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            var confirm = MessageBox.Show(
                this,
                $"Uninstall package '{package}'?",
                "Confirm uninstall",
                MessageBoxButtons.YesNo,
                MessageBoxIcon.Warning);

            if (confirm != DialogResult.Yes)
                return;

            try
            {
                var res = env.Run($"-m pip uninstall -y {package}");
                if (res.ExitCode != 0)
                {
                    MessageBox.Show(this, res.StdErr, "Uninstall failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }

                RefreshPackages();
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, ex.Message, "Uninstall failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
    }
}
