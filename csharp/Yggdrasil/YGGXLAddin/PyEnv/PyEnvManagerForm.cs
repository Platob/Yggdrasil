using System;
using System.Drawing;
using System.Windows.Forms;
using YGGXLAddin.PyEnv;

namespace YGGXLAddin
{
    public sealed class PyEnvManagerForm : Form
    {
        private readonly PyEnvManager _manager;
        private readonly ListView _envList;
        private readonly TextBox _nameInput;
        private readonly TextBox _versionInput;
        private readonly Label _baseDirLabel;
        private readonly Button _refreshButton;
        private readonly Button _createButton;
        private readonly Button _deleteButton;

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
                RowCount = 4,
                Padding = new Padding(12)
            };

            layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
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

            _versionInput = new TextBox { Dock = DockStyle.Fill, Text = "3.11.6" };
            createPanel.Controls.Add(_versionInput, 3, 0);

            _createButton = new Button
            {
                Text = "Create",
                AutoSize = true
            };
            _createButton.Click += (_, __) => CreateEnvironment();
            createPanel.Controls.Add(_createButton, 4, 0);

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
            layout.Controls.Add(actionPanel, 0, 3);

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

            if (!PyVersion.TryParse(versionText, out var version))
            {
                MessageBox.Show(this, "Provide a version like 3.11.6.", "Invalid version", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            try
            {
                _manager.Create(name, version);
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
    }
}
