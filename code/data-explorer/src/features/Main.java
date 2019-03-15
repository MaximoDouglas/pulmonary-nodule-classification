package features;

import features.LungImagesDB;

public class Main
{

	public static void main(String[] args) 
	{
		System.out.println("RUNNING...");
		
		LungImagesDB db = new LungImagesDB("db_tcc");
		db.makeCSVfile_solidNodules("/home/douglas/dev/tcc/features/features_solidNodules");
		
		System.out.println("FINISHED");
	}
	
}

