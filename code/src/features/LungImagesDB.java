package features;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.UnknownHostException;
import com.mongodb.BasicDBList;
import com.mongodb.BasicDBObject;
import com.mongodb.DB;
import com.mongodb.DBCollection;
import com.mongodb.DBCursor;
import com.mongodb.MongoClient;

import features.FeaturesNames;

public class LungImagesDB
{

	private DB db;
	private DBCollection col;

	public static final String NODULE_IMAGE = "noduleImage";
	public static final String PARENCHYMA_IMAGE = "parenchymaImage";
	public static final String ORIGINAL_IMAGE = "originalImage";

	private void connectDB(String databaseName)
	{
		try
		{
			MongoClient mongoClient = new MongoClient("127.0.0.1", 27017);
			db = mongoClient.getDB(databaseName);
			col = db.getCollection("exams");

		} catch (UnknownHostException e)
		{
			System.err.println("Erro ao tentar se conectar ao banco " + databaseName);
			e.printStackTrace();
		} 
	}

	public LungImagesDB(String databaseName)
	{
		this.connectDB(databaseName);
	}

	/**
	 * Método que seleciona os documentos de uma dada coleção.
	 * @param collection
	 */
	public void loadCollection(String collection)
	{
		if(db.collectionExists(collection))	col = db.getCollection(collection);
	}

	public void makeCSVfile_solidNodules(String pathName){
		try
		{
			pathName += ".csv";
			File f = new File(pathName);
			FileWriter fw;
			fw = new FileWriter(f);
			BufferedWriter bw = new BufferedWriter(fw);

			String header = new String();

			header = "id,";

			for(int i = 0; i < FeaturesNames.intensityAttributesNames_nodule.length; ++i)
				header += FeaturesNames.intensityAttributesNames_nodule[i] + ",";

//			for(int i = 0; i < FeaturesNames.intensityAttributesNames_parenchyma.length; ++i)
//				header += FeaturesNames.intensityAttributesNames_parenchyma[i] + ",";


			for(int i = 0; i < FeaturesNames.shapeAttributesNames.length; ++i)
				header += FeaturesNames.shapeAttributesNames[i] + ",";

			for(int i = 0; i < FeaturesNames.textureAttributesNames_nodule.length; ++i)
				header += FeaturesNames.textureAttributesNames_nodule[i] + ",";

//			for(int i = 0; i < FeaturesNames.textureAttributesNames_parenchyma.length; ++i)
//				header += FeaturesNames.textureAttributesNames_parenchyma[i] + ",";

			for(int i = 0; i < FeaturesNames.marginSharpnessNames.length; ++i)
				header += FeaturesNames.marginSharpnessNames[i] + ",";

			header += "diameter_mm,malignancy,class";
			bw.write(header);
			bw.newLine();

			this.loadCollection("exams");
			DBCursor cursor = col.find().sort(new BasicDBObject("path",1));

			BasicDBObject exam;
			BasicDBObject reading;
			BasicDBObject nodule;

			BasicDBList noduleList;
			BasicDBList attributeList;

			while(cursor.hasNext())
			{
				exam = (BasicDBObject) cursor.next();
				reading = (BasicDBObject) exam.get("readingSession");
				noduleList = (BasicDBList) reading.get("bignodule");
				
				

				for (int i_nodule = 0; i_nodule < noduleList.size(); i_nodule++)
				{
					nodule = (BasicDBObject) noduleList.get(i_nodule);
					String texture = (String) nodule.get("texture");

					double d = (double) nodule.get("diameter");
					
					if(nodule.containsField("marginAttributes3D") && nodule.containsField("textureAttributes") 
							&& (d >= 3) && (d <= 30) && texture.equals("5") && !nodule.get("malignancy").equals("3")){
						header = new String();
						
						String id = exam.getObjectId("_id").toString() + "#";
						
						header = id + ",";

						attributeList = (BasicDBList) nodule.get("noduleIntensityAttributes3D");
						for (int i_attribute = 0; i_attribute < attributeList.size(); i_attribute++) 
						{
							header += Double.toString((double) attributeList.get(i_attribute)) + ",";
						}

//						attributeList = (BasicDBList) nodule.get("parenchymaIntensityAttributes3D");
//						for (int i_attribute = 0; i_attribute < attributeList.size(); i_attribute++) 
//						{
//							header += Double.toString((double) attributeList.get(i_attribute)) + ",";
//						}

						attributeList = (BasicDBList) nodule.get("noduleShapeAttributes");
						for (int i_attribute = 0; i_attribute < attributeList.size(); i_attribute++) 
						{
							header += Double.toString((double) attributeList.get(i_attribute)) + ",";
						}

						attributeList = (BasicDBList) nodule.get("textureAttributes");
						for (int i_attribute = 0; i_attribute < attributeList.size(); i_attribute++)
						{
							header += (String) attributeList.get(i_attribute) + ",";
						}

//						attributeList = (BasicDBList) nodule.get("parenchymaTextureAttributes3D");
//						for (int i_attribute = 0; i_attribute < attributeList.size(); i_attribute++)
//						{
//							header += (String) attributeList.get(i_attribute) + ",";
//						}

						attributeList = (BasicDBList) nodule.get("marginAttributes3D");
						for (int i_attribute = 0; i_attribute < attributeList.size(); i_attribute++)
						{
							if(i_attribute != 0) header += Double.toString((double) attributeList.get(i_attribute)) + ",";
							else 								 header += Integer.toString((int) attributeList.get(i_attribute)) + ",";
						}

						double diameter = (double) nodule.get("diameter");
						header += Double.toString(diameter) + ",";

						String malignancy = (String) nodule.get("malignancy");
						header += malignancy + ",";

						String _class = new String();
						if(malignancy.equals("1") || malignancy.equals("2")) 			_class = "BENIGN";
						else if(malignancy.equals("4") || malignancy.equals("5")) _class = "MALIGNANT";
						header += _class;

						bw.write(header.toString());
						bw.newLine();
					}
				}
			}
			
			bw.close();
			fw.close();
		} catch (IOException e)
		{
			System.err.println("Erro! Não foi possível criar o arquivo " + pathName);
			e.printStackTrace();
		}
	}
}
