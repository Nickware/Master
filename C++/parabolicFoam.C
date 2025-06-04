#include "fvCFD.H"

// Función para exportar el campo U a CSV
void exportVelocityToCSV(const volVectorField& U, const fvMesh& mesh)
{
    fileName outputDir = "octaveData";
    mkDir(outputDir);
    
    OFstream os(outputDir/"velocityData.csv");
    
    // Encabezado CSV
    os << "x,y,z,Ux,Uy,Uz" << endl;
    
    // Datos de todas las celdas
    forAll(U, celli)
    {
        const vector& center = mesh.C()[celli];
        const vector& vel = U[celli];
        os << center.x() << "," << center.y() << "," << center.z() 
           << "," << vel.x() << "," << vel.y() << "," << vel.z() << endl;
    }
}

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    Info<< "Creando campo de velocidad U\n" << endl;

    // Crear campo de velocidad (como en tu ejemplo original)
    volVectorField U
    (
        IOobject
        (
            "U",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedVector("U", dimensionSet(0, 1, -1, 0, 0), vector(1, 0, 0))
    );

    // Asignar un perfil parabólico en la dirección x (como en tu ejemplo)
    forAll(U, celli)
    {
        const scalar y = mesh.C()[celli].y();
        const scalar h = 0.1;  // altura del canal
        U[celli].x() = 1.0 - sqr((y - h/2)/(h/2));
    }

    // Exportar datos a Octave
    exportVelocityToCSV(U, mesh);
    
    U.write();

    Info<< "Datos exportados para visualización en Octave\n" << endl;

    return 0;
}
