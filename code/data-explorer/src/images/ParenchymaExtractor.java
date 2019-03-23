package images;

import java.awt.Color;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.Vector;

import org.bson.types.ObjectId;

import com.mongodb.gridfs.GridFS;

import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.Roi;
import ij.process.AutoThresholder;
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
		
		//testes
//		mask.show();
//		IJ.save("/home/ailton/Dropbox/workspace.java/Pulmonary.Nodules.Project/imgs/" + imageName + "-Mask-M1.jpg");
	}
	
	/**
	 * Carrega na memória uma imagem de TC completa.
	 * @param imagePath - caminho da imagem com a extensão.
	 */
	public ParenchymaExtractor(String imagePath)
	{
		image = new ImagePlus(imagePath);
		mask = null;
		
//	image.show();
	}
	
	/**
	 * Carrega na memória uma imagem de TC completa.
	 * @param image - imagem de TC.
	 */
	public ParenchymaExtractor(ImagePlus image)
	{
		this.image = image;
		mask = null;
		
//	image.show();
	}
	
	/**
	 * Carrega na memória uma imagem de TC completa.
	 * @param imagePath - caminho da imagem com a extensão.
	 */
	public void readImage(String imagePath)
	{
		image = new ImagePlus(imagePath);
		mask = null;
		
//		image.show();
	}
	
	/**
	 * Método que ajusta a janela de nível de cinza para o pulmão
	 */
	public void ajustLevelWindow()
	{
		CT_Window_Level windowLevel = new CT_Window_Level();
		windowLevel.run(image);
//		windowLevel.run("Test");
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
		
//		roi.show();
//		image.show();
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

		/*if(parenchymaImage.getType() == ImagePlus.GRAY16) System.out.println("16");
		else if(parenchymaImage.getType() == ImagePlus.GRAY8) System.out.println("8");*/ 
		
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
		
		//TODO tirar 
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
	 * Método para testes
	 */
	public void trash()
	{
		ImagePlus imageTemp;
		ImagePlus mask;
		
		BufferedImage bufferedImage;
		ImageProcessor processor;
		String imageName = new String("sample1"); 
		
		//Lendo imagem
//		imageTemp = new ImagePlus("/home/ailton/Dropbox/workspace.java/Pulmonary.Nodules.Project/imgs/" + imageName + ".jpg"); //home
		imageTemp =  new ImagePlus("/home/felix/Dropbox/workspace.java/Pulmonary.Nodules.Project/imgs/sample2/original2.tc.dcm");
//		image = new ImagePlus("/home/felix/Dropbox/workspace.java/Pulmonary.Nodules.Project/imgs/" + imageName + ".jpg");	//lab

		/*newImage = new ImagePlus();
		bufferedImage = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
		newImage.setImage(bufferedImage);*/
		
//		int k[] = image.getPixel(0, 0);
//		System.out.println("Antes: " + k[0] + ", " + k[1] + ", " + k[2] + ", " + k[3]);
		
		// ----------------- Método 1 ----------------- 
		mask = imageTemp.duplicate();
		IJ.setAutoThreshold(mask, "Huang dark"); //aplicando limiazização
		IJ.run(mask, "Convert to Mask", "");	//criando uma máscara com o parênquima em branco
		mask.show();
//		IJ.save("/home/ailton/Dropbox/workspace.java/Pulmonary.Nodules.Project/imgs/" + imageName + "-Mask-M1.jpg");
		
		// ----------------- Método 2 ----------------- 
		/*image = new ImagePlus("/home/ailton/Dropbox/workspace.java/Pulmonary.Nodules.Project/imgs/" + imageName + ".jpg"); //home
		IJ.setAutoThreshold(image, "MinError dark"); //aplicando limiazização
		IJ.run(image, "Convert to Mask", "");	//criando uma máscara com o parênquima em branco
		image.show();
		IJ.save("/home/ailton/Dropbox/workspace.java/Pulmonary.Nodules.Project/imgs/" + imageName + "-Mask-M2.jpg");*/

		
		//Aplicando o threshold
//		processor = image.getProcessor();
//		processor.setAutoThreshold(AutoThresholder.Method.Huang, true); 	   //M1
//		processor.setAutoThreshold(AutoThresholder.Method.MinError, true); 	 //M2
//		IJ.save("/home/felix/Dropbox/workspace.java/Pulmonary.Nodules.Project/imgs/" + imageName + "-M1.jpg");
//		k = image.getPixel(0, 0);
//		System.out.println(image.getType());
//		System.out.println("Depois: " + k[0] + ", " + k[1] + ", " + k[2] + ", " + k[3]);
		
		//#####################################################################
		//Convertendo imagem para escala de cinza
		/*BufferedImage resultImage = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
		resultImage.getGraphics().drawImage(image.getBufferedImage(), 0, 0, null);
		
		int rgb[] = image.getPixel(0, 0);
		System.out.println(rgb[0] + ", " + rgb[1]);
		
		int argb = resultImage.getRGB(0, 0);
		int a = (argb>>24)&0xff;
		int r = (argb>>16)&0xff;
		int g = (argb>>8)&0xff;
		int b = argb&0xff;
		System.out.println(argb);
		System.out.println("a = " + a + "\n (" + r + ", " + g + ", " + b + ")");
		
		for (int j = 0; j < resultImage.getHeight(); ++j)
		{
			for (int i = 0; i < resultImage.getWidth(); ++i)
			{
			}
		}*/
		//#####################################################################
		
//		image.setImage(resultImage);
//		image.show();
//		int i[] = image.getPixel(0, image.getWidth()-1);		 
		
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
