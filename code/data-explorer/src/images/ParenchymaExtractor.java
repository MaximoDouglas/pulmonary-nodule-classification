package images;

import ij.IJ;
import ij.ImagePlus;
import ij.process.ImageConverter;
import ij.process.ImageProcessor;

public class ParenchymaExtractor
{
	private ImagePlus image;
	private ImagePlus roi;
	private ImagePlus mask;

	/**
	 * Métodos de limiarização
	 */
	public static String HUANG = "Huang dark";
	public static String MINERROR = "MinError dark";

	/**
	 * Salva uma imagem em disco. 
	 * @param image - imagem carregada na memória
	 * @param pathName - localização + nome da imagem.
	 * @param format - extensão da imagem.
	 */
	public static void saveImage(ImagePlus image, String pathName, String format)
	{
		IJ.saveAs(image, format, pathName);
	}

	/**
	 * Retorna uma nova imagem de novo tamanho centralizada.
	 * @param image - imagem original.
	 * @param newWidth - nova largura da imagem.
	 * @param newHeight - nova altura da imagem.
	 * @return - imagem com background redimensionado.
	 */
	public static ImagePlus resizeCanvas(ImagePlus image, int newWidth, int newHeight)
	{
		ImageProcessor processor = image.getProcessor();
		ImageProcessor newProcessor = processor.createProcessor(newWidth, newHeight);
		newProcessor.setColor(0); //preenchendo a nova imagem com preto
		newProcessor.fill();
		int xOff = (newWidth - image.getWidth()) / 2;
		int yOff = (newHeight - image.getHeight()) / 2;	  
		newProcessor.insert(processor, xOff, yOff);

		return new ImagePlus("temp", newProcessor);
	}

	/**
	 * Cria e carrega na memória uma máscara com a região do parênquima em branco.
	 * @param method - método de limiarização
	 */
	private void createMask(String method)
	{
		mask = roi.duplicate();
		IJ.setAutoThreshold(mask, method);
		IJ.run(mask, "Convert to Mask", "");	//criando uma máscara com o parênquima em branco
	}

	/**
	 * Carrega na memória uma imagem de TC completa.
	 * @param imagePath - caminho da imagem com a extensão.
	 */
	public ParenchymaExtractor(String imagePath)
	{
		image = new ImagePlus(imagePath);
		mask = null;
	}

	/**
	 * Carrega na memória uma imagem de TC completa.
	 * @param image - imagem de TC.
	 */
	public ParenchymaExtractor(ImagePlus image)
	{
		this.image = image;
		mask = null;
	}

	/**
	 * Carrega na memória uma imagem de TC completa.
	 * @param imagePath - caminho da imagem com a extensão.
	 */
	public void readImage(String imagePath)
	{
		image = new ImagePlus(imagePath);
		mask = null;
	}

	/**
	 * Método que ajusta a janela de nível de cinza para o pulmão
	 */
	public void ajustLevelWindow()
	{
		CT_Window_Level windowLevel = new CT_Window_Level();
		windowLevel.run(image);
	}

	/**
	 * Criar a roi a partir de uma imagem de tomografia computadorizada
	 * @param x
	 * @param y
	 * @param width
	 * @param height
	 */
	public void cutRoi(int x, int y, int width, int height)
	{
		ImageProcessor processor = image.getProcessor();
		processor.setRoi(x, y, width, height);
		roi = new ImagePlus("roi", processor.crop());
	}

	/**
	 * Método que extrai a região do parênquima de uma roi.
	 * @param method - método de limiarização
	 * @return imagem roi contendo a região do parênquima com outras estruturas do pulmão na cor preta.
	 */
	public ImagePlus run(String method)
	{
		ImagePlus parenchymaImage = roi.duplicate();

		//Convertendo a imagem para 8-bits (256 cores)
		ImageConverter converter = new ImageConverter(parenchymaImage);
		converter.convertToGray8(); 

		createMask(HUANG);	//criando a máscara

		int columns = mask.getWidth();
		int lines = mask.getHeight();

		//varredura coluna inteira -> linha
		for (int j = 0; j < lines; ++j)
		{
			for (int i = 0; i < columns; ++i)
			{
				int pixel[] = mask.getPixel(i, j); //usar apenas o indíce 0 para imagens em escala de cinza.

				if(pixel[0] == 255) //região preta, ruído
				{
					pixel[0] = 0; //escolhendo a cor preta
					ImageProcessor processor = parenchymaImage.getProcessor();
					processor.putPixel(i, j, pixel);
				}
			}
		}

		saveImage(image, "original", "png");
		saveImage(roi, "roi", "png");
		saveImage(mask, "mask", "png");
		saveImage(parenchymaImage, "parenchyma", "png");

		return parenchymaImage;
	}

	public ImagePlus getImage()
	{
		return image;
	}

	public ImagePlus getRoi()
	{
		return roi;
	}

	public ImagePlus getMask()
	{
		return mask;
	}

	public void showImage()
	{
		image.show("Original Imaga");
	}

	public void closeImage()
	{
		image.close();
	}

	public void showRoi()
	{
		roi.show("Roi");
	}

	public void closeRoi()
	{
		roi.close();
	}

	public void showMask()
	{
		mask.show("Mask");
	}

	public void closeMask()
	{
		mask.close();
	}

	/**
	 * Método que converte a imagem para escala de cinza 8 bits
	 */
	public ImagePlus convert8Bits(ImagePlus parenchymaImage)
	{
		ImageConverter converter = new ImageConverter(parenchymaImage);
		converter.convertToGray8();
		ImagePlus parenchymaImage8Bits = parenchymaImage;

		return parenchymaImage8Bits;
	}
}
