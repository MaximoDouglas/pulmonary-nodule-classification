package images;

import java.awt.*;
import java.awt.event.*;
import ij.*;
import ij.process.*;
import ij.gui.*;
import ij.plugin.frame.*;
import java.util.Vector;


/** This plugin allows the window and level of a 16-bit grayscale image or stack
 * (typical CT scan data) to be set or selected from presets. 
 */

@SuppressWarnings("serial")
public class CT_Window_Level extends PlugInFrame implements Runnable,
ActionListener, AdjustmentListener, ItemListener {

	public static final int BONE_W=2000, BONE_L=300;
	public static final int ABDO_W=350, ABDO_L=50;
	public static final int LIVER_W=150, LIVER_L=80;
	public static final int LUNG_W=1600, LUNG_L=-600;
	public static final int SINUS_W=4000, SINUS_L=400;
	public static final int BRAIN_W=80, BRAIN_L=40;
	public static final int STROKE_W=30, STROKE_L=30;
	public static final int SDH_W=130, SDH_L=65;
	public static final int SOFT_W=350, SOFT_L=0;
	public static final int CTA_W=450, CTA_L=150;


	public static final String LOC_KEY = "ctwandl.loc";
	static final int AUTO_THRESHOLD = 5000;
	static final String[] presetLabels = {"None", "Full range", "Image", "Bone", "Abdo", "Liver", "Lung", "Sinus", "Brain", "Stroke", "Subdural", "Soft tissue", "CT Angio" };
	static final String dcmLEVEL="0028,1050", dcmWINDOW="0028,1051", dcmINTERCEPT="0028,1052", dcmSLOPE="0028,1053", dcmPIXREP="0028,0103";
	static final int[] presetWindows = {BONE_W,ABDO_W,LIVER_W,LUNG_W,SINUS_W,BRAIN_W,STROKE_W,SDH_W,SOFT_W,CTA_W};
	static final int[] presetLevels = {BONE_L,ABDO_L,LIVER_L,LUNG_L,SINUS_L,BRAIN_L,STROKE_L,SDH_L,SOFT_L,CTA_L};

	Thread thread;
	private static Frame instance;

	int levelValue=-10000, windowValue=-10000;
	double slope = 1;
	double intercept = -1024;
	int sliderRange = 256;
	boolean doReset,doSet,doWrite;
	Panel panel, tPanel;
	Button resetB, setB, writeB;
	ImageJ ij;
	double defaultMin, defaultMax;
	int offSigned;
	int fullWindow, fullLevel;
	int imgWindow=-99999, imgLevel=-99999;
	int window=BONE_W, level=BONE_L;
	int preset;
	Scrollbar windowSlider, levelSlider;
	Label windowLabel, levelLabel, nullLabel;
	boolean done;
	GridBagLayout gridbag;
	GridBagConstraints c;
	int y = 0;
	Font monoFont = new Font("Monospaced", Font.PLAIN, 12);
	Font sanFont = new Font("SansSerif", Font.PLAIN, 12);
	Choice choice;
	private int windowRange, levelMax=5000, levelMin=-1000, windowMax=10000;
	//	static Vector<WinLevel> winlevelList = new Vector<WinLevel>();
	Vector<WinLevel> winlevelList = new Vector<WinLevel>();


	public CT_Window_Level() {
		super("CT W&L");
	}


	public void run(String arg) {
		setTitle("CT W&L");


		if (instance!=null) {
			if (!instance.getTitle().equals(getTitle())) {
				CT_Window_Level ca = (CT_Window_Level)instance;
				Prefs.saveLocation(LOC_KEY, ca.getLocation());
				ca.close();
			} else {
				instance.toFront();
				return;
			}
		}
		instance = this;
		IJ.register(CT_Window_Level.class);
		WindowManager.addWindow(this);

		ij = IJ.getInstance();
		gridbag = new GridBagLayout();
		c = new GridBagConstraints();
		c.fill = GridBagConstraints.BOTH;
		setLayout(gridbag);

		// level slider
		levelSlider = new Scrollbar(Scrollbar.HORIZONTAL, level, 1, levelMin, levelMax);
		c.gridy = y++;
		c.insets = new Insets(12, 10, 0, 10);
		gridbag.setConstraints(levelSlider, c);
		add(levelSlider);
		levelSlider.addAdjustmentListener(this);
		levelSlider.addKeyListener(ij);
		levelSlider.setUnitIncrement(1);
		levelSlider.setFocusable(false);

		addLabel("Level (Center) (HU): ", levelLabel=new TrimmedLabel("        "));

		// window slider
		windowSlider = new Scrollbar(Scrollbar.HORIZONTAL, window, 1, 0, windowMax);
		c.gridy = y++;
		c.insets = new Insets(2, 10, 0, 10);
		c.fill = GridBagConstraints.BOTH;
		gridbag.setConstraints(windowSlider, c);
		add(windowSlider);
		windowSlider.addAdjustmentListener(this);
		windowSlider.addKeyListener(ij);
		windowSlider.setUnitIncrement(1);
		windowSlider.setFocusable(false);

		addLabel("Window (HU):            ", windowLabel=new TrimmedLabel("        "));

		c.gridy = y++;
		c.insets = new Insets(5, 10, 5, 10);
		choice = new Choice();
		for (int i=0; i<presetLabels.length; i++)
			choice.addItem(presetLabels[i]);
		gridbag.setConstraints(choice, c);
		choice.addItemListener(this);
		add(choice);

		// buttons
		int trim = IJ.isMacOSX()?20:0;
		panel = new Panel();
		panel.setLayout(new GridLayout(0,3, 3, 0));

		resetB = new TrimmedButton("Reset",trim);
		resetB.addActionListener(this);
		panel.add(resetB);

		setB = new TrimmedButton("Set",trim);
		setB.addActionListener(this);
		panel.add(setB);

		writeB = new TrimmedButton("Write",trim);
		writeB.addActionListener(this);
		panel.add(writeB);

		c.gridy = y++;
		c.insets = new Insets(8, 5, 10, 5);
		gridbag.setConstraints(panel, c);
		add(panel);

		addKeyListener(ij);  // ImageJ handles keyboard shortcuts

		pack();

		Point loc = Prefs.getLocation(LOC_KEY);
		if (loc!=null)
			setLocation(loc);
		else
			GUI.center(this);
		if (IJ.isMacOSX()) setResizable(false);
		this.setVisible(true);

		thread = new Thread(this, "CT Window and Level");
		thread.start();
		setup();
	}


	/**
	 * by Ailton
	 */
	void run(ImagePlus imp) 
	{
		//		Roi roi = imp.getRoi();
		//		if (roi!=null) roi.endPaste();
		ImageProcessor ip = imp.getProcessor();
		if(imgProcessed(imp)<0) setupNewImage(imp, ip);
	}

	void addLabel(String text, Label label2) {
		if (label2==null&&IJ.isMacOSX()) text += "    ";
		panel = new Panel();
		c.gridy = y++;

		int bottomInset = IJ.isMacOSX()?4:0;
		c.insets = new Insets(0, 8, bottomInset+4, 8);
		gridbag.setConstraints(panel, c);
		panel.setLayout(new FlowLayout(label2==null?FlowLayout.CENTER:FlowLayout.LEFT, 0, 0));
		Label label= new TrimmedLabel(text);
		label.setFont(sanFont);
		panel.add(label);
		if (label2!=null) {
			label2.setFont(monoFont);
			label2.setAlignment(Label.RIGHT);
			label2.setMinimumSize(new Dimension(60,20));
			panel.add(label2);
		}

		add(panel);
	}

	void setup() {
		ImagePlus imp = WindowManager.getCurrentImage();
		WinLevel curr_wl = new WinLevel();
		if(imp==null) return;
		ImageProcessor ip = imp.getProcessor();

		if (imgProcessed(imp)<0) {
			if(imp.getType()!=ImagePlus.GRAY16) {
				IJ.error("Invalid Type", "16-bit grayscale image or stack required");
				return;
			}
			setup(imp);
			updateLabels(imp);
			imp.updateAndDraw();
			return;
		}
		if(imgProcessed(imp)>=0 && imp.getType()==ImagePlus.GRAY16){
			curr_wl=winlevelList.get(imgProcessed(imp));
			slope=curr_wl.getSlope();
			intercept=curr_wl.getIntercept();
			window=curr_wl.getWindow();
			level=curr_wl.getLevel();
			imgWindow=curr_wl.getimgWindow();
			imgLevel=curr_wl.getimgLevel();
			fullWindow=curr_wl.getfullWindow();
			fullLevel=curr_wl.getfullLevel();
			offSigned=curr_wl.getoffSigned();
			preset=curr_wl.getPreset();
			adjustLevel(imp, ip, level);
			adjustWindow(imp, ip, window);
			if(preset>=0) choice.select(preset);
			else {
				choice.select("None");
				preset = 0;
			}

			updateLabels(imp);
			imp.updateAndDraw();
			return;
		}
	}

	public synchronized void adjustmentValueChanged(AdjustmentEvent e) {
		Object source = e.getSource();
		if (source==windowSlider)
			window = windowValue = windowSlider.getValue();
		else
			level = levelValue = levelSlider.getValue();
		notify();
	}

	public synchronized  void actionPerformed(ActionEvent e) {
		Button b = (Button)e.getSource();
		if (b==null) return;
		if (b==resetB)
			doReset = true;
		else if (b==setB)
			doSet = true;
		else if (b==writeB)
			doWrite = true;
		notify();
	}

	ImageProcessor setup(ImagePlus imp) {
		Roi roi = imp.getRoi();
		if (roi!=null) roi.endPaste();
		ImageProcessor ip = imp.getProcessor();
		if(imgProcessed(imp)<0) setupNewImage(imp, ip);

		imp.updateAndDraw();

		return ip;
	}

	int imgProcessed(ImagePlus imp){
		//returns the index of the current window if it has been processed;
		int index=-1;
		if(imp==null) return index;
		for(WinLevel wl : winlevelList) {
			if(wl.getWin()==imp.getWindow()){
				index=winlevelList.indexOf(wl);
			}
		}
		return index;
	}

	void setupNewImage(ImagePlus imp, ImageProcessor ip)  {
		int size = imp.getStackSize();
		int pixelrep=0;
		int value, slicemax, slicemin, slicesize=imp.getWidth()*imp.getHeight();
		double dicomtag;
		short[] pixels;
		ImageProcessor sp;

		defaultMin = 65535;
		defaultMax = 0;
		for (int slice=1; slice<=size; slice++) {
			sp = imp.getStack().getProcessor(slice);
			pixels = (short[]) sp.getPixels();
			slicemin = 65535;
			slicemax = 0;
			for (int i=0; i<slicesize; i++) {
				value = pixels[i]&0xffff;
				if (value<slicemin) slicemin = value;
				if (value>slicemax) slicemax = value;
			}
			if (slicemin<defaultMin) defaultMin = slicemin;
			if (slicemax>defaultMax) defaultMax = slicemax;
		}

		//Get Slope
		dicomtag=getDicomNumTag(imp, dcmSLOPE);
		if (dicomtag!=-99999) slope=dicomtag;
		else slope=1;								//best guess
		//Get Intercept
		dicomtag=getDicomNumTag(imp, dcmINTERCEPT);
		if (dicomtag!=-99999) intercept=dicomtag;
		else intercept=-1024;						//best guess
		//Get Image Level
		dicomtag=getDicomNumTag(imp, dcmLEVEL);
		if (dicomtag!=-99999) imgLevel=LUNG_L;//(int)dicomtag;
		else imgLevel=LUNG_L;
		//Get Image Window
		dicomtag=getDicomNumTag(imp, dcmWINDOW);
		if (dicomtag!=-99999) imgWindow=LUNG_W;//(int)dicomtag;
		else imgWindow=LUNG_W;
		//Get Pixel representation
		dicomtag=getDicomNumTag(imp, dcmPIXREP);		//0 => unsigned, 1 => signed
		if (dicomtag!=-99999) pixelrep=(int)dicomtag;
		else pixelrep=0;

		offSigned = (32768 + (int)intercept)*pixelrep;          //bug fix to give correct offset

		defaultMax-=offSigned;
		defaultMin-=offSigned;

		fullWindow = (int)((defaultMax-defaultMin)*slope);
		fullLevel = (int) (ScrtoHU(defaultMin) + fullWindow / 2);
		if(imgWindow!=-99999 && imgLevel!=-99999) {
			window = imgWindow;
			level = imgLevel;
			//			choice.select("Image");
			preset=2;
		} else {
			window = fullWindow;
			level = fullLevel;
			//			choice.select("Full range");
			preset=1;
		}
		if(imgProcessed(imp)<0) {
			WinLevel newwl =new WinLevel(imp.getWindow(), slope, intercept, window, level, imgWindow, imgLevel, fullWindow, fullLevel, offSigned, 2);
			winlevelList.add(newwl);
		}
		adjustLevel(imp, ip, level);
		adjustWindow(imp, ip, window);
	}

	void setMinAndMax(ImagePlus imp, double min, double max) {
		if(imp.getType()!=ImagePlus.GRAY16) {
			IJ.error("Invalid Type", "16-bit grayscale image or stack required");
			return;
		}
		imp.setDisplayRange(min, max);
	}



	void updateLabels(ImagePlus imp) {
		windowLabel.setText(""+window);
		levelLabel.setText(""+level);

	}

	void updateScrollBars(Scrollbar sb, boolean newRange) {
		if (sb==null || sb!=windowSlider) {
			if (windowSlider!=null) {
				if (newRange)
					windowSlider.setValues(window, 1, 0,  windowRange);
				else
					windowSlider.setValue(window);
			}
		}
		if (sb==null || sb!=levelSlider) {
			if (newRange)
				levelSlider.setValues(level, 1, 0,  sliderRange);
			else
				levelSlider.setValue(level);
		}

	}


	void adjustLevel(ImagePlus imp, ImageProcessor ip, double lvalue) {
		int scrLevel = (int)HUtoScr(lvalue)+offSigned;
		int scrWindow = (int)(window/slope);
		double min = scrLevel - scrWindow / 2;
		double max = scrLevel + scrWindow / 2;
		setMinAndMax(imp, min, max);
		saveWinLevel(imp);
		//		updateScrollBars(levelSlider, false);
	}

	void adjustWindow(ImagePlus imp, ImageProcessor ip, int wvalue) {
		int scrLevel = (int)HUtoScr(level)+offSigned;
		int scrWindow = (int)(wvalue/slope);

		double min=scrLevel-scrWindow/2;
		double max=scrLevel+scrWindow/2;
		setMinAndMax(imp, min, max);
		saveWinLevel(imp);
		//		updateScrollBars(windowSlider, false);
	}

	void reset(ImagePlus imp, ImageProcessor ip) {
		if(imgWindow!=-99999 && imgLevel!=-99999) {
			window = imgWindow;
			level = imgLevel;
			choice.select("Image");
		} else {
			window = fullWindow;
			level = fullLevel;
			choice.select("Full range");
		}
		adjustLevel(imp, ip, level);
		adjustWindow(imp, ip, window);

	}

	void setWindowLevel(ImagePlus imp, ImageProcessor ip) {		
		GenericDialog gd = new GenericDialog("Set W&L");
		gd.addNumericField("Window Center (Level): ", level, 0);
		gd.addNumericField("Window Width: ", window, 0);

		gd.showDialog();
		if (gd.wasCanceled())
			return;
		level = (int) gd.getNextNumber();
		window = (int) gd.getNextNumber();

		adjustLevel(imp, ip, level);
		adjustWindow(imp, ip, window);
		choice.select("None");
		preset=0;
		saveWinLevel(imp);
	}

	static final int RESET=0, SET=1, LEVEL=2, WINDOW=3, WRITE=4;

	// Separate thread that does the potentially time-consuming processing
	public void run() {
		while (!done) {
			synchronized(this) {
				try {wait();}
				catch(InterruptedException e) {}
			}
			doUpdate();
		}
	}

	void doUpdate() {
		ImagePlus imp;
		ImageProcessor ip;
		int action;

		if (doReset) action = RESET;
		else if (doSet) action = SET;
		else if (doWrite) action = WRITE;
		else if (levelValue!=-10000) action = LEVEL;
		else if (windowValue!=-10000) action = WINDOW;
		else return;
		levelValue = windowValue = -10000;
		doReset = doSet = doWrite = false;
		imp = WindowManager.getCurrentImage();
		if (imp==null) {
			IJ.beep();
			IJ.showStatus("No image");
			return;
		}
		ip = imp.getProcessor();

		switch (action) {
		case RESET: reset(imp, ip); break;
		case SET: setWindowLevel(imp, ip); break;
		case LEVEL: adjustLevel(imp, ip, level); break;
		case WINDOW: adjustWindow(imp, ip, window); break;
		case WRITE: writeHeader(imp); break;
		}

		updateLabels(imp);

		imp.updateChannelAndDraw();

	}


	public void windowClosing(WindowEvent e) {
		close();
		Prefs.saveLocation(LOC_KEY, getLocation());
	}

	/** Overrides close() in PlugInFrame. */

	public void close() {
		super.close();
		instance = null;
		done = true;
		synchronized(this) {
			notify();
		}
	}


	@SuppressWarnings("unlikely-arg-type")
	public void windowActivated(WindowEvent e) {
		super.windowActivated(e);
		Vector<Integer> closedList = new Vector<Integer>(); //list of closed windows
		for(WinLevel wl: winlevelList){
			if(wl.getWin().isClosed()) {
				closedList.add(winlevelList.indexOf(wl));
			}
		}
		
		for(Integer i: closedList) {
			winlevelList.remove(i);
		}
		
		closedList.clear();

		ImagePlus imp = WindowManager.getCurrentImage();
		int storedImg = imgProcessed(imp);
		if(imp!=null && imp.getType()==ImagePlus.GRAY16) setup();
		if(storedImg>=0 && imp.getType()!=ImagePlus.GRAY16) {		//if the image type has been changed
			IJ.error("Image type no longer valid", "16-bit grayscale image or stack required");
			winlevelList.remove(getWinLevel(imp));
		}
		
		WindowManager.setWindow(this);
	}

	public synchronized  void itemStateChanged(ItemEvent e) {
		int index = choice.getSelectedIndex();

		switch(index) {
		case 0:
			break;
		case 1:
			level=levelValue=fullLevel;
			window=windowValue=fullWindow;
			break;
		case 2:
			if(imgWindow!=-99999 && imgLevel!=-99999) {
				window = windowValue = imgWindow;
				level = levelValue = imgLevel;
				choice.select("Image");
				preset=2;
			} else {
				window = levelValue = fullWindow;
				level = windowValue = fullLevel;
				choice.select("Full range");
				preset=1;
			}
			break;
		default:
			preset=index;
			level=levelValue=presetLevels[preset-3];
			window=windowValue=presetWindows[preset-3];
			break;
		}
		notify();
	}

	/** Resets this and brings it to the front. */
	public void updateAndDraw() {
		toFront();
	}


	double getDicomNumTag(ImagePlus imp, String tag) {
		double tagval=-99999;
		String str_tagval, header;
		ImageStack stack = imp.getStack();
		int stk_size=stack.getSize();
		if(stk_size>1) header = stack.getSliceLabel(imp.getCurrentSlice());
		else header = (String)imp.getProperty("Info");
		if(header!=null){
			int iTag = header.indexOf(tag);
			int iColon = header.indexOf(":",iTag);
			int iNewline = header.indexOf("\n",iColon);
			if(iTag>=0 && iColon>=0 && iNewline>=0){
				str_tagval = header.substring(iColon+1,iNewline).trim();
				tagval=Double.parseDouble(str_tagval);
			}
		}
		return tagval;
	}

	void writeHeader(ImagePlus imp) {
		if(!isDicomWL(imp)) {
			IJ.error("Format problem", "No DICOM Window and Level tags to write to");
			return;
		}
		if(!IJ.showMessageWithCancel("CT Window & Level", "Write to image header?")) return;
		ImageStack stack = imp.getStack();
		String header;
		int stk_size=stack.getSize();
		for(int s=1; s<=stk_size; s++) {
			if(stk_size>1) header = stack.getSliceLabel(imp.getCurrentSlice());
			else header = (String)imp.getProperty("Info");
			StringBuffer hdrBuff = new StringBuffer(header);
			hdrBuff = writeinttoTag(hdrBuff, dcmLEVEL, level);
			hdrBuff = writeinttoTag(hdrBuff, dcmWINDOW, window);
			String new_header = new String(hdrBuff);
			if(stk_size>1) stack.setSliceLabel(new_header, s);
			else imp.setProperty("Info", new_header);
		}
		imgWindow=window;
		imgLevel=level;
		saveWinLevel(imp);
	}

	private boolean isDicomWL(ImagePlus imp) {
		String header;
		ImageStack stack = imp.getStack();
		int stk_size=stack.getSize();
		if(stk_size>1) header = stack.getSliceLabel(imp.getCurrentSlice());
		else header = (String)imp.getProperty("Info");
		if(header.indexOf(dcmWINDOW)<0 || header.indexOf(dcmLEVEL)<0) return false;
		return true;
	}

	StringBuffer writeinttoTag(StringBuffer hdrBuff, String tag, int value) {
		int iTag = hdrBuff.indexOf(tag);
		int iColon = hdrBuff.indexOf(":",iTag);
		int iNewline = hdrBuff.indexOf("\n",iColon);
		String valtext = new String(" "+value);
		if(iTag>=0 && iColon>=0 && iNewline>=0){
			hdrBuff.replace(iColon+1, iNewline, valtext);
		}
		return hdrBuff;
	}

	double HUtoScr (double HUValue) {
		return (HUValue-intercept)/slope;
	}

	double HUtoScr (int HUValue) {
		return (HUValue-intercept)/slope;
	}

	double ScrtoHU (double ScrValue) {
		return ScrValue*slope+intercept;
	}

	double ScrtoHU (int ScrValue) {
		return ScrValue*slope+intercept;
	}

	private WinLevel getWinLevel(ImagePlus imp) {
		for(WinLevel wl: winlevelList) {
			if(wl.getWin()==imp.getWindow()) return wl;
		}
		return null;
	}

	void saveWinLevel(ImagePlus imp) {
		WinLevel save_wl = new WinLevel(imp.getWindow(), slope, intercept, window, level, imgWindow, imgLevel, fullWindow, fullLevel, offSigned, preset);
		int index = 0;
		for(WinLevel wl: winlevelList) {
			if(wl.getWin()==imp.getWindow()){
				index = winlevelList.indexOf(wl);
				break;
			}
		}
		winlevelList.set(index, save_wl);
	}

}
@SuppressWarnings("serial")
class TrimmedLabel extends Label {
	int trim = IJ.isMacOSX()?0:6;

	public TrimmedLabel(String title) {
		super(title);
	}


	public Dimension getMinimumSize() {
		return new Dimension(super.getMinimumSize().width, super.getMinimumSize().height-trim);
	}


	public Dimension getPreferredSize() {
		return getMinimumSize();
	}

}

class WinLevel {
	private int window, level, imgWindow, imgLevel, fullWindow, fullLevel, offSigned=0, preset=-1;
	private double slope, intercept;
	private ImageWindow win;

	WinLevel(ImageWindow win, double slope, double intercept, int window, int level, int imgWindow, int imgLevel, int fullWindow, int fullLevel, int offSigned, int preset) {
		this.win=win;
		this.slope=slope;
		this.intercept=intercept;
		this.window=window;
		this.level=level;
		this.imgWindow=imgWindow;
		this.imgLevel=imgLevel;
		this.fullWindow=fullWindow;
		this.fullLevel=fullLevel;
		this.offSigned=offSigned;
		this.preset=preset;
	}

	WinLevel(WinLevel wl) {
		this.win=wl.win;
		this.slope=wl.slope;
		this.intercept=wl.intercept;
		this.window=wl.window;
		this.level=wl.level;
		this.imgWindow=wl.imgWindow;
		this.imgLevel=wl.imgLevel;
		this.fullWindow=wl.fullWindow;
		this.fullLevel=wl.fullLevel;
		this.offSigned=wl.offSigned;
		this.preset=wl.preset;
	}

	WinLevel() {
		return;
	}


	public ImageWindow getWin() {
		return win;
	}

	public int getWindow() {
		return window;
	}

	public double getSlope() {
		return slope;
	}

	public double getIntercept() {
		return intercept;
	}

	public int getLevel() {
		return level;
	}

	public int getoffSigned() {
		return offSigned;
	}

	public int getimgWindow() {
		return imgWindow;
	}

	public int getimgLevel() {
		return imgLevel;
	}

	public int getfullWindow() {
		return fullWindow;
	}

	public int getfullLevel() {
		return fullLevel;
	}

	public int getPreset() {
		return preset;
	}
}


