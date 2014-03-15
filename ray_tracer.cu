/*
 * Realtime Ray Tracer
 *
 * References: (Had to learn this from somewhere)
 *  http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtrace0.htm
 *  http://www.cs.unc.edu/~rademach/xroads-RT/RTarticle.html
 *  http://fuzzyphoton.tripod.com/howtowrt.htm
 *  http://www.visualization.hpc.mil/wiki/Raytracing
 *
 * Revision History:
 *    Mar 07 2012    Nnoduka Eruchalu    Initial Revision
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// for parsing
#include <fstream>
#include <sstream>
#include <iostream>
#include <assert.h>

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// CUDA utilities and system includes
#include <cuda_gl_interop.h>
#include <cutil_inline.h>

// type defines come in handy
typedef unsigned int uint;
typedef unsigned char uchar;

// include ray tracer kernel functions for ease of compiling
#include <tracer_kernel.cu>

/* Screen drawing kernel parameters */
uint width = 512, height = 512;
dim3 blockSize(8, 8);
dim3 gridSize(width / blockSize.x, height / blockSize.y);

/* View settings */
float3 viewRotation = make_float3(0.5, 0.5, 0.0);
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];

GLuint pbo = 0;     // OpenGL pixel buffer object

void initPixelBuffer();

// render image using CUDA -- execute volume rendering kernel
void render()
{
    // set necessary constants in hardware
    cutilSafeCall( cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix,
                                      sizeof(float4)*3) );
    
    // map PBO to get CUDA device pointer
    uint *d_output;
    cutilSafeCall(cudaGLMapBufferObject((void**)&d_output, pbo));
    cutilSafeCall(cudaMemset(d_output, 0, width*height*4));
    
    // call CUDA kernel, writing results to PBO
    d_render<<< gridSize, blockSize >>>(d_output, width, height);
    
    cutilCheckMsg("kernel failed");
    
    cutilSafeCall(cudaGLUnmapBufferObject(pbo));
}

// display results using OpenGL (called by GLUT)
void display()
{
    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
       glLoadIdentity();
       glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
       glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
       glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    // transpose matrix to conform with OpenGL's notation
    invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4];
    invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5];
    invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6];
    invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];
  
    render();
    
    // display results
    glClear(GL_COLOR_BUFFER_BIT);
    
    // draw image from PBO
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    
    glutSwapBuffers();
    glutReportErrors();
}

void idle()
{
}


/* parse line in scene input file
 *
 * args:   line = line froms scene input file
 *         object = true for object, false for light source
 *         object_index = index into object array
 *         light_index = index into light array
 *
 * returns: true for good parse
 *          false for error
 */
bool parseLine(std::string line, bool & is_object, int & object_index,
              int & light_index)
{
    using namespace std; 
    // property
    string prop;
    
    // property values
    string type;
    double cx,cy,cz;
    double r;
    double ambx,amby,ambz,ambw, diffx,diffy,diffz,diffw,specx,specy,specz,specw;
    double shiny,n,kr,kt;
    double minx,miny,minz, maxx,maxy,maxz;
    double ymin,ymax;
    double plx,ply,plz,plw;
    double posx, posy, posz;
    
    // check indices:
    if (object_index >= NUM_OBJECTS) {
        cout << "too many objects in scene file... max allowed is: "
             << NUM_OBJECTS << endl;
        return false;
    }
    if (light_index >= NUM_LIGHTS) {
        cout << "too many light sources in scene file... max allowed is: "
             << NUM_LIGHTS << endl;
        return false;
    }
    
    // parsing -- crude but simple!
    if (line.empty())
        return true;
    stringstream ss(stringstream::in | stringstream::out);
    ss.str(line);
    ss >> prop;
    // access strings as arrays
    if (prop[0] == '#') { // comments
        return 1;
    } else if(prop[0] == '*') { // end of file indicator
        cout << "read entire file successfully" << endl;
        return true;
                
    } else if (prop.compare("type")==0) {   
        ss >> type;
        if (type != "LIGHT") { // if not light then an object
            is_object = true; 
            object_index++;    // new object!
        } else {
            is_object = false;
            light_index++;     // new light source
        }
        // record object types
        if (type== "SPHERE")
            object[object_index].type = SPHERE;
         if (type== "BOX")
             object[object_index].type = BOX;
         if (type== "PLANE")
             object[object_index].type = PLANE;
         if (type== "CYLINDER")
             object[object_index].type = CYLINDER;
         if (type== "CONE")
             object[object_index].type = CONE;
         
    } else if (prop.compare("c")==0) {
        ss >> cx >> cy >>cz; // now you have an x,y,z as floats
        object[object_index].c = make_float3(cx,cy,cz);
        
    } else if (prop.compare("r")==0) {
        ss >> r;
        object[object_index].r = r;
    
    } else if (prop.compare("amb")==0) {
        ss >> ambx >> amby >> ambz >> ambw;
        if (is_object)
            object[object_index].mat.amb = make_float4(ambx, amby, ambz, ambw);
        else
            light[light_index].amb = make_float4(ambx, amby, ambz, ambw);
        
    } else if (prop.compare("diff")==0) {
        ss >> diffx >> diffy >> diffz >> diffw;
        if (is_object)
            object[object_index].mat.diff=make_float4(diffx,diffy,diffz,diffw);
        else
            light[light_index].diff = make_float4(diffx,diffy,diffz,diffw);
        
    } else if (prop.compare("spec")==0) {
        ss >> specx >> specy >> specz >> specw;
        if (is_object)
            object[object_index].mat.spec=make_float4(specx,specy,specz,specw);
        else
             light[light_index].spec = make_float4(specx,specy,specz,specw);
            
    } else if (prop.compare("shiny")==0) {
        ss >> shiny;
        object[object_index].mat.shiny = shiny;
    
    } else if (prop.compare("n")==0) {
        ss >> n;
        object[object_index].mat.n = n;
    
    } else if (prop.compare("kr")==0) {
        ss >> kr;
        object[object_index].mat.kr = kr;
    
    } else if (prop.compare("kt")==0) {
        ss >> kt;
        object[object_index].mat.kt = kt;
    
    } else if (prop.compare("min")==0) {
        ss >> minx >> miny >> minz;
        object[object_index].min = make_float3(minx, miny, minz);
        
    } else if (prop.compare("max")==0) {
        ss >> maxx >> maxy >> maxz;
        object[object_index].max = make_float3(maxx, maxy, maxz);
        
    } else if (prop.compare("pl")==0) {
        ss >> plx >> ply >> plz >> plw;
        object[object_index].pl = make_float4(plx, ply, plz, plw);
        
    } else if (prop.compare("ymin")==0) {
        ss >> ymin;
        object[object_index].ymin = ymin;
        
    }  else if (prop.compare("ymax")==0) {
        ss >> ymax;
        object[object_index].ymax = ymax;
        
    } else if (prop.compare("pos")==0) { // for light objects only
        ss >> posx >> posy >>posz;
        light[light_index].pos = make_float3(posx,posy,posz);
        
    } else {
        cout << "property " << prop << " doesn't exit\nquitting now...\n";
        return false;
    }
    
    
    if (ss.fail()) {
        return false;
    }
        
    return true;
}

// parse scene input file
void parseScene(std::string filename)
{
    using namespace std;
    char line[1024]; // create some temporary storage
    bool is_object = true;
    int object_index = -1; // parseLine automatically incrememnts
    int light_index = -1;
    ifstream inFile(filename.c_str(), ifstream::in); // open as stream
    if (!inFile) {
        cout << "could not open given file " << filename << endl;
        exit(1);
    }
    while (inFile.good()) {
        inFile.getline(line,1023); // read line into temporary storage
        // parse the line
        if (!parseLine(string(line), is_object, object_index, light_index)) {
            exit(1);                 // an error occurred?
        }    
    }
    inFile.close(); // always do this
}

void initScene(int argc, char** argv)
{
    using namespace std;
    // read the command line
    if (argc != 2) {
        cout << "usage: " << argv[0] << " scene.in" << endl;
        exit(1);
    }
        
    parseScene(std::string(argv[1]));         // setup objects in scene
    bgnd_col = make_float4(0.5f); // grey background

    cutilSafeCall( cudaMemcpyToSymbol(d_object, &object, sizeof(object)) );
    cutilSafeCall( cudaMemcpyToSymbol(d_light,  &light,  sizeof(light)) );
    cutilSafeCall( cudaMemcpyToSymbol(d_bgnd_col, &bgnd_col, sizeof(float4)) );
}

void keyboard(unsigned char key, int x, int y)
{
    switch(key)
    {
    case 27:
        exit(0);
    }
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;
    
    ox = x; oy = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - ox;
    dy = y - oy;
  
    if (buttonState == 4)
    {   // right = zoom
        viewTranslation.z += dy / 100.0;
    } 
    else if (buttonState == 2)
    {   // middle = translate
        viewTranslation.x += dx / 100.0;
        viewTranslation.y -= dy / 100.0;
    }
    else if (buttonState == 1)
    {   // left = rotate
        viewRotation.x += dy / 5.0;
        viewRotation.y += dx / 5.0;
    }
    
    ox = x; oy = y;
    glutPostRedisplay();
}

void reshape(int x, int y)
{
    width = x; height = y;
    // reinitialize with new size
    initPixelBuffer();
    
    glViewport(0, 0, x, y);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}


void cleanup()
{
    cutilSafeCall(cudaGLUnregisterBufferObject(pbo));    
    glDeleteBuffersARB(1, &pbo);
}

int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void initPixelBuffer()
{
    if (pbo)
    {
        // delete old buffer
        cutilSafeCall(cudaGLUnregisterBufferObject(pbo));
        glDeleteBuffersARB(1, &pbo);
    }
    
    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4,
                    0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    
    cutilSafeCall(cudaGLRegisterBufferObject(pbo));
    
    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
    // use command-line specified CUDA device, otherwise use device with
    // highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );
    
    int device;
    struct cudaDeviceProp prop;
    cudaGetDevice( &device );
    cudaGetDeviceProperties( &prop, device );
    if( !strncmp( "Tesla", prop.name, 5 ) )
    {
        printf("This sample needs a card capable of OpenGL and display.\n");
        printf("Please choose a different device with the "
               "-device=x argument.\n");
        cutilExit(argc, argv);
    }
    
    printf("Mouse CLicks Legend: Right  = zoom\n"
           "                     Middle = translate\n"
           "                     Left   = rotate\n");
    
    // initialize GLUT callback functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width*2, height*2);
    glutCreateWindow("CUDA Realtime Ray Tracing - Nnoduka Eruchalu");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);
    
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"))
    {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(-1);
    }
    initPixelBuffer();
    
    initScene(argc, argv); // initialize Scene to be rendered
    
    atexit(cleanup);
    
    glutMainLoop();
    
    cudaThreadExit();
    return 0;
}
