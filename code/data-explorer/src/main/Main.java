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
		
		Features features = new Features(db);
		features.makeCSVfile_solidNodules("/home/douglas/dev/tcc/data/features/");
		features.makeCSVfile_solidNodules_withParenchyma("/home/douglas/dev/tcc/data/features/");
		
		Segmented_Solid_Nodules ssn_images = new Segmented_Solid_Nodules(db);
		ssn_images.downloadImages("/home/douglas/dev/tcc/data/images/solid-nodules/");
		
		Segmented_Solid_Nodules_With_Attributes ssnwa_images = new Segmented_Solid_Nodules_With_Attributes(db);
		ssnwa_images.downloadImages("/home/douglas/dev/tcc/data/images/solid-nodules-with-attributes/");
		
		System.out.println("FINISHED");
	}
	
}

