package images;

import com.mongodb.BasicDBList;
import com.mongodb.DB;
import com.mongodb.DBCollection;
import com.mongodb.BasicDBObject;
import com.mongodb.DBCursor;
import com.mongodb.gridfs.GridFS;
import com.mongodb.gridfs.GridFSDBFile;
import com.mongodb.gridfs.GridFSInputFile;

import ij.ImagePlus;
import ij.process.ImageConverter;
import ij.process.ImageProcessor;

import java.awt.Polygon;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import org.bson.types.ObjectId;

public class Images {

	private DB db;	
	private double step = 0.05;

	public Images(DB db) {
		this.db = db;
	}

	public void downloadImages_solidNodules(String rootPath, String tag) throws IOException {

		rootPath += "solid-nodules" + "/";

		try{

			//GenerateImages object will store the Dicom images from the database
			GenerateImages getImage = new GenerateImages(db);

			BasicDBObject exam;
			BasicDBObject reading;
			BasicDBObject bignodule = null;
			BasicDBObject roi = null;

			BasicDBList bignoduleList;
			BasicDBList roiList = null;

			DBCollection collection = db.getCollection("exams");
			DBCursor cursor = collection.find();

			int examCount = 0, nodulesCount = 0, notUsedNodulesCount = 0, 
					benignNodulesCount = 0, malignantNodulesCount = 0;

			while(cursor.hasNext()){

				examCount++;
				exam = (BasicDBObject) cursor.next();

				reading = (BasicDBObject) exam.get("readingSession");

				bignoduleList = (BasicDBList) reading.get("bignodule");

				for (int i_nodule = 0; i_nodule < bignoduleList.size(); i_nodule++){

					bignodule = (BasicDBObject) bignoduleList.get(i_nodule);

					nodulesCount++;

					Double diameter = (Double) bignodule.get("diameter");
					roiList = (BasicDBList) bignodule.get("roi");
					String malignancyNumber, malignancyName = "";
					String texture = (String) bignodule.get("texture");
					malignancyNumber = (String) bignodule.get("malignancy");

					double d = diameter.doubleValue();

					if (!malignancyNumber.equals("3") && (d >= 3) && (d <= 30) && texture.equals("5")) {

						if(malignancyNumber.equals("1") || malignancyNumber.equals("2")){
							malignancyName = "benigno";

							benignNodulesCount++;

							File diretorio = new File(rootPath + malignancyName + "/" + benignNodulesCount + "/");
							diretorio.mkdir();

						} else if(malignancyNumber.equals("4") || malignancyNumber.equals("5")){
							malignancyName = "maligno";

							malignantNodulesCount++;

							File diretorio = new File(rootPath + malignancyName + "/" + malignantNodulesCount + "/");
							diretorio.mkdir();

						}

						for (int i_roi = 0; i_roi < roiList.size(); ++i_roi) {
							roi = (BasicDBObject) roiList.get(i_roi);
							String fileNameP1 = "", fileNameP2 = "";

							if(malignancyName.equals("benigno") ){

								fileNameP1 = rootPath + malignancyName + "/" + benignNodulesCount + "/"; 
								fileNameP2 = malignancyName + benignNodulesCount + "-" + i_roi;		

								getImage.generateImage(roi.getObjectId(tag), fileNameP1 + fileNameP2, ".png");

							} else if(malignancyName.equals("maligno") ){

								fileNameP1 = rootPath + malignancyName + "/" + malignantNodulesCount + "/"; 
								fileNameP2 = malignancyName +  malignantNodulesCount + "-" + i_roi;		

								getImage.generateImage(roi.getObjectId(tag), fileNameP1 + fileNameP2, ".png");

							}
						}
					} else {
						notUsedNodulesCount++;
					}
				} 
			}

			System.out.println("Terminou a geração das imagens Dicom");
			System.out.println("Total de exames: " + examCount);			
			System.out.println("Total de nódulos: " + nodulesCount);
			System.out.println("Nodulos Benignos: " + benignNodulesCount);
			System.out.println("Nodulos Malignos: " + malignantNodulesCount);
			System.out.println("Nodulos não utilizados: " + notUsedNodulesCount);

		} catch(Exception e){
			e.printStackTrace();
			System.err.println( e.getClass().getName() + ": " + e.getMessage() );
		}
	}

	public void downloadImages_solidNodules_withAttributes(String rootPath, String tag) throws IOException {

		rootPath += "solid-nodules-with-attributes" + "/";

		try{

			//GenerateImages object will store the Dicom images from the database
			GenerateImages getImage = new GenerateImages(db);

			BasicDBObject exam;
			BasicDBObject reading;
			BasicDBObject bignodule = null;
			BasicDBObject roi = null;

			BasicDBList bignoduleList;
			BasicDBList roiList = null;

			DBCollection collection = db.getCollection("exams");
			DBCursor cursor = collection.find();

			int examCount = 0, nodulesCount = 0, notUsedNodulesCount = 0, 
					benignNodulesCount = 0, malignantNodulesCount = 0;

			while(cursor.hasNext()){

				examCount++;
				exam = (BasicDBObject) cursor.next();
				String exam_id = exam.getObjectId("_id").toString();

				reading = (BasicDBObject) exam.get("readingSession");

				bignoduleList = (BasicDBList) reading.get("bignodule");

				for (int i_nodule = 0; i_nodule < bignoduleList.size(); i_nodule++){

					bignodule = (BasicDBObject) bignoduleList.get(i_nodule);
					String nodule_id = bignodule.get("noduleID").toString();

					nodulesCount++;

					Double diameter = (Double) bignodule.get("diameter");
					roiList = (BasicDBList) bignodule.get("roi");
					String malignancyNumber, malignancyName = "";
					String texture = (String) bignodule.get("texture");
					malignancyNumber = (String) bignodule.get("malignancy");

					double d = diameter.doubleValue();

					if (bignodule.containsField("marginAttributes3D") && bignodule.containsField("textureAttributes") 
							&& !malignancyNumber.equals("3") && (d >= 3) && (d <= 30) && texture.equals("5")) {

						if(malignancyNumber.equals("1") || malignancyNumber.equals("2")){
							malignancyName = "benigno";

							benignNodulesCount++;

							String examFolder = rootPath + malignancyName + "/" + exam_id + "/";

							File directory = new File(examFolder);
							directory.mkdir();

							directory = new File(examFolder + nodule_id + "/");
							directory.mkdir();

						} else if(malignancyNumber.equals("4") || malignancyNumber.equals("5")){
							malignancyName = "maligno";

							malignantNodulesCount++;

							String examFolder = rootPath + malignancyName + "/" + exam_id + "/";

							File directory = new File(examFolder);
							directory.mkdir();

							directory = new File(examFolder + nodule_id + "/");
							directory.mkdir();

						}

						for (int i_roi = 0; i_roi < roiList.size(); ++i_roi) {
							roi = (BasicDBObject) roiList.get(i_roi);
							String fileNameP1 = "", fileNameP2 = "";

							if(malignancyName.equals("benigno") ){

								fileNameP1 = rootPath + malignancyName + "/" + exam_id + "/" + nodule_id + "/"; 
								fileNameP2 = String.valueOf(i_roi);		

								getImage.generateImage(roi.getObjectId(tag), fileNameP1 + fileNameP2, ".png");

							} else if(malignancyName.equals("maligno") ){

								fileNameP1 = rootPath + malignancyName + "/" + exam_id + "/" + nodule_id + "/"; 
								fileNameP2 = String.valueOf(i_roi);		

								getImage.generateImage(roi.getObjectId(tag), fileNameP1 + fileNameP2, ".png");

							}
						}
					} else {
						notUsedNodulesCount++;
					}
				} 
			}

			System.out.println("Terminou a geração das imagens Dicom");
			System.out.println("Total de exames: " + examCount);			
			System.out.println("Total de nódulos: " + nodulesCount);
			System.out.println("Nodulos Benignos: " + benignNodulesCount);
			System.out.println("Nodulos Malignos: " + malignantNodulesCount);
			System.out.println("Nodulos não utilizados: " + notUsedNodulesCount);

		} catch(Exception e){
			e.printStackTrace();
			System.err.println( e.getClass().getName() + ": " + e.getMessage() );
		}
	}

	/**
	 * Recupera uma imagem no formato ImagePlus a partir do GridFS do banco. 
	 * @param originalImage - chave da imagem no banco
	 * @param imageType - formato da imagem
	 * @return
	 */
	private ImagePlus restoreImage(ObjectId originalImage, String imageType)
	{
		try
		{
			GridFS fileStore;
			GridFSDBFile gridFile;
			ImagePlus dicom;
			String fileName;

			fileName = "./temp." + imageType;

			fileStore = new GridFS(db, "images");

			gridFile = fileStore.find(originalImage);
			gridFile.writeTo(fileName);

			dicom = new ImagePlus(fileName);

			return dicom;
		} catch (IOException e)
		{
			System.err.println("Erro! Não foi possível salvar o arquivo de imagem");
			e.printStackTrace();
		}

		return null;
	}

	/**
	 * Escreve na tela o percentual de progresso de uma execução interativa.
	 * @param current - valor atual da interação.
	 * @param total - Valor total da interação.
	 */
	public void progress(int current, int total){
		if(current == (int)(step*total))
		{
			System.out.println(step*100.0 + "%");
			step = step + 0.05;
		}
	}

	public void writeOnDB_NoduleImages_CTWindow() throws IOException{
		
		DBCollection col = db.getCollection("exams"); //carregando a coleção dos exames.

		/*Mongo mongoClient = new Mongo( "127.0.0.1" , 27017 );
		// To connect to mongodb server
		//MongoClient m = new MongoClient( "localhost" , 27017 );

		// Now connect to your databases, nome da base aqui foi exams
		DB db = mongoClient.getDB( "exams" );
		System.out.println("Connect to database successfully");

		// conecta a coleção exams
		DBCollection collection = db.getCollection("exams");
		 */
		DBCursor cursor = col.find();//.sort(new BasicDBObject("path",1)); //recuperando os documento(s) da coleção
		cursor.addOption(com.mongodb.Bytes.QUERYOPTION_NOTIMEOUT);

		ImagePlus parenchymaImage;
		ImagePlus parenchymaImage2;

		BasicDBObject exam;
		BasicDBObject reading;
		BasicDBObject bignodule;
		BasicDBObject roi;
		BasicDBObject edgeMap; 

		BasicDBList bignoduleList;
		BasicDBList roiList;
		BasicDBList edgeMapList;
		BasicDBList edgeMapList2;

		int min_x, max_x;
		int min_y, max_y;
		int maxDiameter_x;
		int maxDiameter_y;

		int examCount = 0;  
		//		while(examCount < 44) {cursor.next(); ++examCount; }//começa no (valor+1) tirar isso aqui
		boolean comment = false;

		//FileWriter arq = new FileWriter("nodulos.txt");
		//PrintWriter gravarArq = new PrintWriter(arq);
		Object id;

		//gravarArq.printf("Exame" + ";" + "Nodulo" + ";" + "i_roi" + ";" + "X" + ";" + "Y" + "%n");


		while(cursor.hasNext()) //para cada exame
		{	
			//if(comment) System.out.println("Exame " + examCount);

			//cursor.count()



			exam = (BasicDBObject) cursor.next();
			id = exam.getObjectId("_id");
			reading = (BasicDBObject) exam.get("readingSession");
			bignoduleList = (BasicDBList) reading.get("bignodule");

			for (int i_nodule = 0; i_nodule < bignoduleList.size(); ++i_nodule) //para cada bignódulo
			{
				String id2;

				if(comment) System.out.println("Bignodulo " + i_nodule);

				bignodule = (BasicDBObject) bignoduleList.get(i_nodule);
				id2 = (String) bignodule.get("noduleID");
				roiList = (BasicDBList) bignodule.get("roi");

				int x1,x2,y1,y2;
				x1 = y1 = 512;
				x2 = y2 = 0;

				for (int i_roi = 0; i_roi < roiList.size(); ++i_roi) //para cada fatia parte2/2
				{
					if(comment) System.out.println("Roi " + i_roi);

					roi = (BasicDBObject) roiList.get(i_roi);
					edgeMapList = (BasicDBList) roi.get("edgeMap");


					for(int i_edgeMap = 0; i_edgeMap < edgeMapList.size(); i_edgeMap++) //para cada coordenada de borda
					{
						edgeMap = (BasicDBObject) edgeMapList.get(i_edgeMap);

						int x = Integer.parseInt((String) edgeMap.get("xCoord"));
						int y = Integer.parseInt((String) edgeMap.get("yCoord"));
						//for(EdgeMap ed : roi.getEdgeMaps()){
						if(x > x2)
							x2 = x;
						if(x < x1)
							x1 = x;
						if(y > y2)
							y2 = y;
						if(y < y1)
							y1 = y;
						//}	

					}

				} //fim para cada roi 

				int width = x2 - x1;
				int height = y2 - y1;

				int dif = 0;
				if(width > height){
					dif = width - height;
					height = width;
					y2 = y2 + dif;
				}
				else{
					dif = height - width;
					width = height;
					x2 = x2 + dif;
				}

				for (int i_roi = 0; i_roi < roiList.size(); ++i_roi) //para cada fatia parte2/2
				{
					if(comment) System.out.println("Roi " + i_roi);

					roi = (BasicDBObject) roiList.get(i_roi);
					//edgeMapList = (BasicDBList) roi.get("edgeMap");


					ParenchymaExtractor extractor = new ParenchymaExtractor(restoreImage(roi.getObjectId("originalImage"), "dcm"));

					//janelamento
					extractor.ajustLevelWindow();

					extractor.convert8Bits(extractor.getImage());
					parenchymaImage = extractor.getImage();//extractor.convert8Bits(extractor.getRoi());//extractor.convert8Bits(extractor.getRoi());
					//parenchymaImage.show();
					//System.out.println(parenchymaImage.getType());

					//BufferedImage buffer = new BufferedImage(width+1,height+1, BufferedImage.TYPE_INT_RGB);
					//buffer = (BufferedImage)parenchymaImage;

					BufferedImage img = new BufferedImage(width+1,height+1, BufferedImage.TYPE_BYTE_GRAY);//TYPE_BYTE_GRAY  );

					ImagePlus imgA = new ImagePlus("imgA2", img);
					//ImagePlus imgA = IJ.createImage("imgA", width+1, height+1, 1, 8);// getImage();


					ImageConverter converter = new ImageConverter(imgA);
					converter.convertToGray8();//convertToRGB();
					//					imgA.show();

					//System.out.println(imgA.getType());


					int xx = 0;
					int yy = 0;

					Polygon region;
					edgeMapList2 = (BasicDBList) roi.get("edgeMap");
					int[] xPoints = new int[edgeMapList2.size()];
					int[] yPoints = new int[edgeMapList2.size()];

					for(int i_edgeMap = 0; i_edgeMap < edgeMapList2.size(); i_edgeMap++) //para cada coordenada de borda
					{
						edgeMap = (BasicDBObject) edgeMapList2.get(i_edgeMap);

						xPoints[i_edgeMap] = Integer.parseInt((String) edgeMap.get("xCoord"));
						yPoints[i_edgeMap] = Integer.parseInt((String) edgeMap.get("yCoord"));
					}

					region = new Polygon(xPoints, yPoints, edgeMapList2.size());

					//parenchymaImage.show();
					for (int x = x1; x <= x2; x++) {
						for (int y = y1; y <= y2; y++) {

							int pixel[] = parenchymaImage.getPixel(x, y); //usar apenas o indíce 0 para imagens em escala de cinza.
							ImageProcessor processor = imgA.getProcessor();
							//int pixel = processor.getPixel(y, x);//  getPixelValue(y, x);
							//System.out.println(pixel);
							if(region.contains(x, y)){
								//ImageProcessor processor = parenchymaImage.getProcessor();
								//int pixel = processor.getPixel(y, x);

								//System.out.println(pixel);
								//img.setRGB(xx, yy, pixel);
								processor.putPixel(xx, yy, pixel);


							}
							else{
								processor.putPixel(xx, yy, 0);
								//img.setRGB(xx, yy, 0);
							}
							yy++;
						}
						yy=0;
						xx++;
					}

					//saveImage(imgA, "imgA2", "png");
					//imgA.show();

					//ImageIO.write(img, "png", new File("//home/lucas/Área de Trabalho/NodulosTestes/"+i_roi+".png"));


					int limit = 256;					
					int x,y;
					x = y = 0;
					BufferedImage img2 = new BufferedImage(limit,limit, BufferedImage.TYPE_BYTE_GRAY);
					ImagePlus imgB = new ImagePlus("imgB2", img2);

					for (int i = 0; i < limit; i++) {
						for (int j = 0; j < limit; j++) {
							ImageProcessor processor2 = imgB.getProcessor();
							if((i < ((limit/2) - (width/2))) || (i > ((limit/2) + (width/2))) || 
									(j < ((limit/2) - (height/2))) || (j > ((limit/2) + (height/2)))){
								processor2.putPixel(i, j, 0);
								//img2.setRGB(i, j, 0);
							}

							else{
								processor2.putPixel(i, j, imgA.getPixel(x, y));
								//img2.setRGB(i, j, img.getRGB(x,y));
								y++;
							}																			
						}
						if(!((i < ((limit/2) - (width/2))) || (i > ((limit/2) + (width/2))))){
							y = 0;
							x++;
						}
					}

					//					imgB.show();

					ParenchymaExtractor.saveImage(imgB, "temp", "png");
					File imageFile =  new File("temp.png");
					GridFS gridFS = new GridFS(db, "images");
					GridFSInputFile gridInputFile = gridFS.createFile(imageFile);
					gridInputFile.setFilename("exam" + examCount + ".n" + i_nodule + ".r" + i_roi + "-noduleImageJT");
					gridInputFile.save();

					//add o id da imagem em 'exams' parte1/3
					GridFSDBFile gridFile = gridFS.findOne("exam" + examCount + ".n" + i_nodule + ".r" + i_roi + "-noduleImageJT");
					roi.append("noduleImageJT", gridFile.getId());
					//roi.remove("noduleImageJ");
					roiList.set(i_roi, roi);


					//ImageIO.write(img2, "png", new File("//home/lucas/Área de Trabalho/NodulosTestes/"+i_roi+".png"));
				} //fim para cada roi 

				//add o id da imagem em 'exams' parte2/3
				bignodule.append("roi", roiList);
				bignoduleList.set(i_nodule, bignodule);
			} //fim para cada nódulo

			//add o id da imagem em 'exams' parte3/3
			reading.append("bignodule", bignoduleList);
			exam.append("readingSession", reading);
			col.update(new BasicDBObject("path", exam.getString("path")), exam);

			progress(examCount, cursor.size());
			++examCount;
		}
	}
}