namespace YGGXLAddin
{
    partial class YGGRibbon : Microsoft.Office.Tools.Ribbon.RibbonBase
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        public YGGRibbon()
            : base(Globals.Factory.GetRibbonFactory())
        {
            InitializeComponent();
        }

        /// <summary> 
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Component Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.YGGAddin = this.Factory.CreateRibbonTab();
            this.group1 = this.Factory.CreateRibbonGroup();
            this.YGGPythonTab = this.Factory.CreateRibbonTab();
            this.groupEnvironment = this.Factory.CreateRibbonGroup();
            this.buttonManageEnvironments = this.Factory.CreateRibbonButton();
            this.YGGAddin.SuspendLayout();
            this.YGGPythonTab.SuspendLayout();
            this.SuspendLayout();
            // 
            // YGGAddin
            // 
            this.YGGAddin.ControlId.ControlIdType = Microsoft.Office.Tools.Ribbon.RibbonControlIdType.Office;
            this.YGGAddin.Groups.Add(this.group1);
            this.YGGAddin.Label = "YGG Addin";
            this.YGGAddin.Name = "YGGAddin";
            // 
            // group1
            // 
            this.group1.Label = "group1";
            this.group1.Name = "group1";
            // 
            // YGGPythonTab
            // 
            this.YGGPythonTab.Groups.Add(this.groupEnvironment);
            this.YGGPythonTab.Label = "YGG Python";
            this.YGGPythonTab.Name = "YGGPythonTab";
            // 
            // groupEnvironment
            // 
            this.groupEnvironment.Items.Add(this.buttonManageEnvironments);
            this.groupEnvironment.Label = "Environment";
            this.groupEnvironment.Name = "groupEnvironment";
            // 
            // buttonManageEnvironments
            // 
            this.buttonManageEnvironments.Label = "Manage Envs";
            this.buttonManageEnvironments.Name = "buttonManageEnvironments";
            this.buttonManageEnvironments.Click += new Microsoft.Office.Tools.Ribbon.RibbonControlEventHandler(this.buttonManageEnvironments_Click);
            // 
            // YGGRibbon
            // 
            this.Name = "YGGRibbon";
            this.RibbonType = "Microsoft.Excel.Workbook";
            this.Tabs.Add(this.YGGAddin);
            this.Tabs.Add(this.YGGPythonTab);
            this.Load += new Microsoft.Office.Tools.Ribbon.RibbonUIEventHandler(this.YGGRibbon_Load);
            this.YGGAddin.ResumeLayout(false);
            this.YGGAddin.PerformLayout();
            this.YGGPythonTab.ResumeLayout(false);
            this.YGGPythonTab.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        internal Microsoft.Office.Tools.Ribbon.RibbonTab YGGAddin;
        internal Microsoft.Office.Tools.Ribbon.RibbonGroup group1;
        internal Microsoft.Office.Tools.Ribbon.RibbonTab YGGPythonTab;
        internal Microsoft.Office.Tools.Ribbon.RibbonGroup groupEnvironment;
        internal Microsoft.Office.Tools.Ribbon.RibbonButton buttonManageEnvironments;
    }

    partial class ThisRibbonCollection
    {
        internal YGGRibbon YGGRibbon
        {
            get { return this.GetRibbon<YGGRibbon>(); }
        }
    }
}
