using Microsoft.Office.Tools.Ribbon;
using System;
using YGGXLAddin.PyEnv;

namespace YGGXLAddin
{
    public partial class YGGRibbon
    {
        private static PyEnvManagerForm _envManagerForm;

        private void YGGRibbon_Load(object sender, RibbonUIEventArgs e)
        {

        }

        private void buttonManageEnvironments_Click(object sender, RibbonControlEventArgs e)
        {
            if (_envManagerForm == null || _envManagerForm.IsDisposed)
            {
                _envManagerForm = new PyEnvManagerForm();
            }

            _envManagerForm.Show();
            _envManagerForm.BringToFront();
        }
    }
}
