package main;

import java.io.IOException;

import com.mongodb.MongoClient;
import com.mongodb.DB;

import features.Features;
import images.Images;

public class Main{

	public static void main(String[] args) throws IOException {
		System.out.println("RUNNING...");
		
		MongoClient mongoClient = new MongoClient("127.0.0.1", 27017);
		DB db = mongoClient.getDB("db_tcc");
		
		String rootPath = "/home/douglas/dev/tcc/data/";
		String featuresPath = rootPath + "features" + "/";
		String imagesPath = rootPath + "images" + "/";
		String windowTag	= "noduleImageJT";
		
		Features features = new Features(db);
		features.makeCSVfile_solidNodules(featuresPath);
		features.makeCSVfile_solidNodules_withParenchyma(featuresPath);
		
		Images images = new Images(db);
		images.downloadImages_solidNodules(imagesPath, windowTag);
		images.downloadImages_solidNodules_withAttributes(imagesPath, windowTag);
		
		System.out.println("FINISHED");
	}
	
}

