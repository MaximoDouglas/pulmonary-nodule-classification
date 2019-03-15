package features;

/**
 * Contém os nomes das features na ordem em que foram armazenados no banco.
 * @author felix
 *
 */
public class FeaturesNames
{
	/**
	 * key -> textureAttributes
	 */
	public final static String[] textureAttributesNames_nodule = {"energy0_N", "entropy0_N", "inertia0_N", "homogeneity0_N", 
			"correlation0_N", "shade0_N", "promenance0_N", "variance0_N", "idm0_N", "energy135_N", "entropy135_N", 
			"inertia135_N", "homogeneity135_N", "correlation135_N", "shade135_N", "promenance135_N", "variance135_N",
			"idm135_N", "energy45_N", "entropy45_N", "inertia45_N", "homogeneity45_N", "correlation45_N", "shade45_N", 
			"promenance45_N", "variance45_N", "idm45_N", "energy90_N", "entropy90_N", "inertia90_N", "homogeneity90_N", 
			"correlation90_N", "shade90_N", "promenance90_N", "variance90_N", "idm90_N"};
	
	/**
	 * key -> parenchymaTextureAttributes3D
	 */
	public final static String[] textureAttributesNames_parenchyma = {"energy0_P", "entropy0_P", "inertia0_P", "homogeneity0_P", 
			"correlation0_P", "shade0_P", "promenance0_P", "variance0_P", "idm0_P", "energy135_P", "entropy135_P", 
			"inertia135_P", "homogeneity135_P", "correlation135_P", "shade135_P", "promenance135_P", "variance135_P",
			"idm135_P", "energy45_P", "entropy45_P", "inertia45_P", "homogeneity45_P", "correlation45_P", "shade45_P", 
			"promenance45_P", "variance45_P", "idm45_P", "energy90_P", "entropy90_P", "inertia90_P", "homogeneity90_P", 
			"correlation90_P", "shade90_P", "promenance90_P", "variance90_P", "idm90_P"};
	
	/**
	 * key -> marginAttributes3D
	 */
	public final static String[] marginSharpnessNames = {"differenceends", "sumvalues", "sumsquares", 
			"sumlogs", "amean", "gmean", "pvariance", "svariance", "sd", "kurtosis", 
			"skewness", "scm"}; 
	
	/**
	 * key -> noduleIntensityAttributes3D 
	 */
	public final static String[] intensityAttributesNames_nodule = {"energy_N", "entropy_N", "kurtosis_N", "maximum_N", "mean_N", 
			"meanAbsoluteDeviation_N", "median_N", "minimum_N", "range_N", "rootMeanSquare_N", "skewness_N", "standardDeviation_N",
			"uniformity_N", "variance_N"};
	
	/**
	 * key -> parenchymaIntensityAttributes3D 
	 */
	public final static String[] intensityAttributesNames_parenchyma = {"energy_P", "entropy_P", "kurtosis_P", "maximum_P", "mean_P", 
			"meanAbsoluteDeviation_P", "median_P", "minimum_P", "range_P", "rootMeanSquare_P", "skewness_P", "standardDeviation_P",
			"uniformity_P", "variance_P"};
	
	/**
	 * key -> noduleShapeAttributes 
	 */
	public final static String[] shapeAttributesNames = {"compactness1", "compactness2", "sphericalDisproportion", 
			"sphericity", "area", "surfaceArea", "surfaceVolumeRatio", "volume", "diameter"};
	
}
