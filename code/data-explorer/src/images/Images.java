package images;

import com.mongodb.BasicDBList;
import com.mongodb.DB;
import com.mongodb.DBCollection;
import com.mongodb.BasicDBObject;
import com.mongodb.DBCursor;
import java.io.File;
import java.io.IOException;

public class Images {
	
	private DB db;
	
	public Images(DB db) {
		this.db = db;
	}
	
	public void downloadImages_solidNodules(String rootPath) throws IOException {
		
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

								getImage.generateImage(roi.getObjectId("noduleImage"), fileNameP1 + fileNameP2, ".png");

							} else if(malignancyName.equals("maligno") ){

								fileNameP1 = rootPath + malignancyName + "/" + malignantNodulesCount + "/"; 
								fileNameP2 = malignancyName +  malignantNodulesCount + "-" + i_roi;		

								getImage.generateImage(roi.getObjectId("noduleImage"), fileNameP1 + fileNameP2, ".png");

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
	
	public void downloadImages_solidNodules_withAttributes(String rootPath) throws IOException {
		
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

								getImage.generateImage(roi.getObjectId("noduleImage"), fileNameP1 + fileNameP2, ".png");

							} else if(malignancyName.equals("maligno") ){

								fileNameP1 = rootPath + malignancyName + "/" + exam_id + "/" + nodule_id + "/"; 
								fileNameP2 = String.valueOf(i_roi);		

								getImage.generateImage(roi.getObjectId("noduleImage"), fileNameP1 + fileNameP2, ".png");

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
}