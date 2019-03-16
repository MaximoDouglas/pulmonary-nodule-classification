package main;

import java.io.IOException;

import com.mongodb.MongoClient;
import com.mongodb.DB;

import features.Features;
import images.Segmented_Solid_Nodules;
import images.Segmented_Solid_Nodules_With_Attributes;

public class Main{

	public static void main(String[] args) throws IOException {
		System.out.println("RUNNING...");
		
		MongoClient mongoClient = new MongoClient("127.0.0.1", 27017);
		DB db = mongoClient.getDB("db_tcc");
		
		String rootPath = "/home/douglas/dev/tcc/data/";
		String featuresPath = rootPath + "features" + "/";
		String imagesPath = rootPath + "images" + "/";
		
		Features features = new Features(db);
		features.makeCSVfile_solidNodules(featuresPath);
		features.makeCSVfile_solidNodules_withParenchyma(featuresPath);
		
		Segmented_Solid_Nodules ssn_images = new Segmented_Solid_Nodules(db);
		ssn_images.downloadImages(imagesPath + "solid-nodules/");
		
		Segmented_Solid_Nodules_With_Attributes ssnwa_images = new Segmented_Solid_Nodules_With_Attributes(db);
		ssnwa_images.downloadImages(imagesPath + "solid-nodules-with-attributes/");
		
		System.out.println("FINISHED");
	}
	
}

