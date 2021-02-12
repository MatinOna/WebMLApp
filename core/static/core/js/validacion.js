function validarExt()
{
    var archivoInput = document.getElementById('archivoInput');
    var archivoRuta = archivoInput.value;
    var extPermitidas = /(.mid|.midi)$/i;

    if(!extPermitidas.exec(archivoRuta))
    {
        alert('El archivo seleccionado no es un archivo compatible (.mid o .midi)');
        archivoInput.value = '';
        return false;
    }
}

function habilitar(value)
{
    if(value=="MLPClassifier")
	{
	    //:)
		document.getElementById("epoch_train").disabled=false;
	}else if(value=="DecisionTree" || value=="RandomForest"){
		//:(
		document.getElementById("epoch_train").disabled=true;
	}
}

	