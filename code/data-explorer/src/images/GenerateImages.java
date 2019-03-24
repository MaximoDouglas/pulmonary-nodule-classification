package images;

import java.io.IOException;
import org.bson.types.ObjectId;
import com.mongodb.DB;
import com.mongodb.gridfs.GridFS;
import com.mongodb.gridfs.GridFSDBFile;

public class GenerateImages {

	DB db;

	public GenerateImages(DB db){
		this.db = db;

	}

	public void generateImage(ObjectId originalImage, String dir, String extension){
		try {
			GridFS fileStore;
			GridFSDBFile gridFile;
			String fileName;
			
			fileName = dir + extension;

			fileStore = new GridFS(db, "images");

			gridFile = fileStore.find(originalImage);
			gridFile.writeTo(fileName);

		} catch (IOException e){
			System.err.println("Erro! Não foi possível salvar o arquivo de imagem");
			e.printStackTrace();
		}
	}
}
