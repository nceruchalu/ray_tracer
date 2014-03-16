/*
 * -----------------------------------------------------------------------------
 * -----                          RAY_TRACER.CU                            -----
 * -----                       REALTIME RAY TRACER                         -----
 * -----------------------------------------------------------------------------
 *
 * File Description:
 *  This is the main loop of the Ray Tracer. It initializes the Pixel Buffer, 
 *  Scene, mouse event handlers and kick-starts scene rendering.
 *
 * Table of Contents:
 *  render          - render image into pixel buffer object. Used by display()
 *  display         - GLUT's display callback for the current window.
 *  idle            - GLUT's global idle callback
 *  keyboard        - GLUT's keyboard callback for the current window.
 *  mouse           - GLUT's mouse callback for the current window
 *  motion          - GLUT's motion callback for the current window.
 *  reshape         - GLUT's reshape callback for the current window.
 *  parseLine       - parse line in input scene description file
 *  parseScene      - parse input scene description file
 *  initScene       - initialize internal scene representation.
 *  cleanup         - used to cleanup memory on program exit
 *  iDivUp          - integer division with implicit ceiling function
 *  initPixelBuffer - initialize Pixel Buffer Object
 *  main            - main loop
 *
 * Assumptions:
 *  Using a machine with a CUDA-capable NVIDIA GPU
 *
 * Limitations:
 *  - Can only initialize scenes using valid scene description files. Syntax is
 *    very specific so see `README.md` or sample `scene.in` files for more 
 *    details
 *
 * References: (Had to learn this from somewhere)
 *  http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtrace0.htm
 *  http://www.cs.unc.edu/~rademach/xroads-RT/RTarticle.html
 *  http://fuzzyphoton.tripod.com/howtowrt.htm
 *  http://www.visualization.hpc.mil/wiki/Raytracing
 *
 * Compiler:
 *  NVIDIA's CUDA Compiler (NVCC)
 *
 * Revision History:
 *  Mar. 07, 2012      Nnoduka Eruchalu     Initial Revision
 *  Mar. 15, 2014      Nnoduka Eruchalu     Cleaned up comments
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// necessary libraries for parsing of input scene description file
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

// type defines that come in handy
typedef unsigned int uint;
typedef unsigned char uchar;

// include ray tracer kernel functions to make compiling easier
#include <tracer_kernel.cu>

/* Screen drawing kernel parameters */
uint width = 512, height = 512;
dim3 blockSize(8, 8);
dim3 gridSize(width / blockSize.x, height / blockSize.y);

/* View settings */
float3 viewRotation = make_float3(0.5, 0.5, 0.0);
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];

/* instantiate and initialize OpenGL's pixel buffer object */
GLuint pbo = 0;   
void initPixelBuffer();

/*
 * render
 * Description:
 *  This procedure renders images (into the pixel buffer object) using the 
 *  CUDA volume rendering kernel.
 *
 * Arguments: 
 *  None
 *
 * Return: 
 *  None
 *
 * Operation:
 *  - set necessary view matrix constants in GPU
 *  - map pixel buffer object into the address space of CUDA. This returns the
 *    base pointer of the resulting mapping.
 *  - clear the values in this newly allocated CUDA address space.
 *  - call CUDA kernel, and write results to pixel buffer object.
 *  - unmap pixel buffer object so CUDA can have access to that memory space.
 *  
 * Revision History:
 *  Mar. 07, 2012      Nnoduka Eruchalu     Initial Revision
 *  Mar. 15, 2014      Nnoduka Eruchalu     Cleaned up comments
 */
void render()
{
    // set necessary constants in hardware
    cutilSafeCall( cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix,
                                      sizeof(float4)*3) );
    
    // map PBO to CUDA device's address space, and get base pointer of the
    // resultant mapping
    uint *d_output;
    cutilSafeCall(cudaGLMapBufferObject((void**)&d_output, pbo));
    cutilSafeCall(cudaMemset(d_output, 0, width*height*4));
    
    // call CUDA kernel, writing results to PBO
    d_render<<< gridSize, blockSize >>>(d_output, width, height);
    
    cutilCheckMsg("kernel failed");
    
    // unmpa PBO to return memory space to CUDA
    cutilSafeCall(cudaGLUnmapBufferObject(pbo));
}


/*
 * display
 * Description:
 *  GLUT's display callback for the current window.
 *
 * Arguments: 
 *  None
 *
 * Return: 
 *  None
 *
 * Operation:
 *  - use OpenGL to build view matrix
 *  - transpaose view matrix to conform with OpenGL's notation
 *  - render image into the pbo (using this view matrix)
 *  - draw image from PBO unto the screen.
 *  
 * Revision History:
 *  Mar. 07, 2012      Nnoduka Eruchalu     Initial Revision
 *  Mar. 15, 2014      Nnoduka Eruchalu     Cleaned up comments
 */
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
    
    // render result into the pbo using this view matrix
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

/*
 * idle
 * Description:
 *  GLUT's global idle callback
 *
 * Arguments: 
 *  None
 *
 * Return: 
 *  None
 *
 * Operation:
 *  - Do Nothing
 *  
 * Revision History:
 *  Mar. 07, 2012      Nnoduka Eruchalu     Initial Revision
 *  Mar. 15, 2014      Nnoduka Eruchalu     Cleaned up comments
 */
void idle()
{
}


/*
 * parseLine
 * Description:
 *  Parse line in scene input file
 *
 * Arguments: 
 *  line         = line froms scene input file
 *  object       = true for object, false for light source
 *  object_index = index into object array
 *  light_index  = index into light array
 *
 * Return: 
 *  true  = successful line parse
 *  false = failed line parse
 *
 * Operation:
 *  - check current number of objects and lights are less than maximums expected
 *    (set in tracer_kernel.cu)
 *  - if line is empty then done and successful, else go on and try to read line
 *  - get first word on line and check for following:
 *    + `#`:     Skip this comment line and return success
 *    + `*`:     Skip this EOF indicator and return success
 *    + `type`:  Get value and increase number of objects/light as appropriate. 
 *               If an object, record object type.
 *    + `c`:     Get x,y,z coordinates for sphere's center and save.
 *    + `r`:     Get object's radius value and record it
 *    + `amb`:   get x,y,z,w coordinates for ambient. 
 *               Save as either object material ambient or light source ambient
 *    + `diff`:  get x,y,z,w coordinates for diffuse. 
 *               Save as either object material's or light source's diffuse
 *    + `spec`:  get x,y,z,w coordinates for specular. 
 *               Save as either object material's or light source's specular
 *    + `shiny`: get object material's shiny and record it.
 *    + `n`:     get object material's refraction index and record it
 *    + `kr`:    get object material's reflective coefficient and record it
 *    + `kt`:    get object material's transmissive coefficient and record it.
 *    + `min`:   get box's minimum vertex value (x,y,z) and record it
 *    + `max`:   get box's maximum vertex value (x,y,z) and record it
 *    + `pl`:    get plane's equation (Ax + By + Cx + D) parameters (A,B,C,D)
 *    + `ymin`:  get object's minimum y-coordinate and record it
 *    + `ymax`:  get object's maximum y-coordinate and record it
 *    + `pos`:   get light source's x,y,z coordinates for position and record it
 *  - if the first word is anything other than what is expected, inform the user
 *    and exit the program.
 *  
 * Revision History:
 *  Mar. 07, 2012      Nnoduka Eruchalu     Initial Revision
 *  Mar. 15, 2014      Nnoduka Eruchalu     Cleaned up comments
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
    
    // check current number of objects and lights are less than max expected
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
    
    /* line parsing commences. This is crude but simple to follow */
    
    // if line is empty, then line parse is done and successful
    if (line.empty())
        return true;
    
    // access strings as arrays of characters
    stringstream ss(stringstream::in | stringstream::out);
    ss.str(line);
    
    // get first word on line and analyze it
    ss >> prop;
    if (prop[0] == '#') { 
        // if first char indicates a comment, then skip line (successful parse)
        return 1;
        
    } else if(prop[0] == '*') {
        // if first char marks end of file, let user know and exit successfully
        cout << "read entire file successfully" << endl;
        return true;
                
    } else if (prop.compare("type")==0) {
        // if first word is the property key `type` grab the next word which
        // will be the value. 
        ss >> type;
        if (type != "LIGHT") { // if not light then an object
            is_object = true; 
            object_index++;    // new object!
        } else {
            is_object = false;
            light_index++;     // new light source
        }
        // record object type
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


/*
 * parseScene
 * Description:
 *  parse scene description input file
 *
 * Arguments: 
 *  filename: name of input file
 *
 * Return: 
 *  None
 *
 * Operation:
 *  - create a buffer to hold the maximum allowed line size
 *  - ensure the file can be opened, if not inform the user and exit
 *  - if file can be opened, read each line into the buffer and call parseLine()
 *  - if any line cannot be successfully parsed, exit the program.
 *  
 * Revision History:
 *  Mar. 07, 2012      Nnoduka Eruchalu     Initial Revision
 *  Mar. 15, 2014      Nnoduka Eruchalu     Cleaned up comments
 */
void parseScene(std::string filename)
{
    using namespace std;
    char line[1024]; // create some temporary storage
    bool is_object = true;
    int object_index = -1; // parseLine automatically incrememnts
    int light_index = -1;
    ifstream inFile(filename.c_str(), ifstream::in); // open as stream
    
    // ensure the file can be opened, else inform the user and exit
    if (!inFile) {
        cout << "could not open given file " << filename << endl;
        exit(1);
    }
    
    // File is good so parse each line. Exit on any failed line parse.
    while (inFile.good()) {
        inFile.getline(line,1023); // read line into temporary storage
        // parse the line
        if (!parseLine(string(line), is_object, object_index, light_index)) {
            exit(1);                 // an error occurred?
        }    
    }
    inFile.close(); // always do this
}


/*
 * initScene
 * Description:
 *  Initialize internal scene representation using provided input scene file.
 *
 * Arguments: 
 *  command line arguments
 *
 * Return: 
 *  None
 *
 * Operation:
 *  - ensure only one argument is provided to this program.
 *  - If 1 argument is given, assume it's scene descriptor and use to setup
 *    internal representation of objects in scene.
 *  
 * Revision History:
 *  Mar. 07, 2012      Nnoduka Eruchalu     Initial Revision
 *  Mar. 15, 2014      Nnoduka Eruchalu     Cleaned up comments
 */
void initScene(int argc, char** argv)
{
    using namespace std;
    // read the command line and ensure only 1 argument is provided
    if (argc != 2) {
        cout << "usage: " << argv[0] << " scene.in" << endl;
        exit(1);
    }
    
    // 1 argument provided so assume it's scene descriptor and setup objects
    parseScene(std::string(argv[1]));
    
    // set background color to grey
    bgnd_col = make_float4(0.5f);

    cutilSafeCall( cudaMemcpyToSymbol(d_object, &object, sizeof(object)) );
    cutilSafeCall( cudaMemcpyToSymbol(d_light,  &light,  sizeof(light)) );
    cutilSafeCall( cudaMemcpyToSymbol(d_bgnd_col, &bgnd_col, sizeof(float4)) );
}


/*
 * keyboard
 * Description:
 *  GLUT's keyboard callback for the current window.
 *
 * Arguments: 
 *  key: keyboard key code of pressed key.
 *  x,y: mouse location in window relative coordinates when key was pressed
 *
 * Return: 
 *  None
 *
 * Operation:
 *  - check if key is <ESC> and if so exit program
 *  
 * Revision History:
 *  Mar. 07, 2012      Nnoduka Eruchalu     Initial Revision
 *  Mar. 15, 2014      Nnoduka Eruchalu     Cleaned up comments
 */
void keyboard(unsigned char key, int x, int y)
{
    switch(key)
    {
    case 27:
        exit(0);
    }
}

/* global variables shared between mouse() and motion() routines */
int ox, oy;          // last recorded x,y positions of mouse
int buttonState = 0; // clicked mouse button state tracker


/*
 * mouse
 * Description:
 *  GLUT's mouse callback for the current window.
 *
 * Arguments: 
 *  button: clicked mouse button (left, middle or right)
 *  state:  indicates mouse button press (GLUT_DOWN) or release (GLUT_UP)
 *  x,y:    mouse location in window relative coords when mouse state changed
 *
 * Return: 
 *  None
 *
 * Operation:
 *  - if mouse was released clear out global mouse state tracker
 *  - if mouse was pressed, OR in the pressed button representation into the
 *    global mouse button state tracker. Note the OR is used to ensure multiple
 *    mouse presses aren't lost.
 *  - record current mouse position in global trackers of mouse position (ox,oy)
 *  - mark current window as needing to be redisplayed.
 *  
 * Revision History:
 *  Mar. 07, 2012      Nnoduka Eruchalu     Initial Revision
 *  Mar. 15, 2014      Nnoduka Eruchalu     Cleaned up comments
 */
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;
    
    ox = x; oy = y;
    glutPostRedisplay();
}


/*
 * motion
 * Description:
 *  GLUT's motion callback for the current window.
 *
 * Arguments: 
 *  x,y:    mouse location in window relative coords
 *
 * Return: 
 *  None
 *
 * Operation:
 *  - determine delta change in mouse coords (dx, dy) since last recording
 *  - The action to take based on these delta depends on clicked mouse button:
 *    + right:  zoom in view using change in y-coord (dy)
 *    + middle: translate view using change in x,y-coords (dx, dy)
 *    + left:   rotate view using change in x,y-coords (dx, dy)
 *  - Now that this motion has been acted on, record these new mouse coords in 
 *    global trackers of mouse position (ox, oy)
 *  - mark current window as needing to be redisplayed.
 *  
 * Revision History:
 *  Mar. 07, 2012      Nnoduka Eruchalu     Initial Revision
 *  Mar. 15, 2014      Nnoduka Eruchalu     Cleaned up comments
 */
void motion(int x, int y)
{
  // determine delta change in mouse position
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


/*
 * reshape
 * Description:
 *  GLUT's reshape callback for the current window.
 *
 * Arguments: 
 *  x: window's new width
 *  y: window's new height
 *
 * Return: 
 *  None
 *
 * Operation:
 *  - set global trackers for window width and height
 *  - reinitialize pixel buffer object and OpenGL with new window parameters
 *  
 * Revision History:
 *  Mar. 07, 2012      Nnoduka Eruchalu     Initial Revision
 *  Mar. 15, 2014      Nnoduka Eruchalu     Cleaned up comments
 */
void reshape(int x, int y)
{
    // set global trackers for window width and height
    width = x; height = y;
    
    // reinitialize PBO and OpenGL with new size
    initPixelBuffer();
    
    glViewport(0, 0, x, y);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}


/*
 * cleanup
 * Description:
 *  cleanup pixel buffer object memory
 *
 * Arguments: 
 *  None
 *
 * Return: 
 *  None
 *
 * Operation:
 *  unregister the pixel buffer object for access by CUDA and releases any CUDA
 *  resources associated with the buffer.
 *  
 * Revision History:
 *  Mar. 07, 2012      Nnoduka Eruchalu     Initial Revision
 *  Mar. 15, 2014      Nnoduka Eruchalu     Cleaned up comments
 */
void cleanup()
{
    cutilSafeCall(cudaGLUnregisterBufferObject(pbo));    
    glDeleteBuffersARB(1, &pbo);
}


/*
 * iDivUp
 * Description:
 *  integer division with implicit ceiling function
 *
 * Arguments: 
 *  a: dividend (numerator)
 *  b: divisor (denominator)
 *
 * Return: 
 *  None
 *
 * Operation:
 *  The goal is to implement this formula: quotient = ceiling(a/b)
 *  - if a/b has a remainder, round up the quotient to the next integer.
 *  - else leave the quotient as is.
 *  Examples: iDivUp(7, 2) = 4  and  iDivUp(6,2) = 3
 *  
 * Revision History:
 *  Mar. 07, 2012      Nnoduka Eruchalu     Initial Revision
 *  Mar. 15, 2014      Nnoduka Eruchalu     Cleaned up comments
 */
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}


/*
 * initPixelBuffer
 * Description:
 *  initialize Pixel Buffer Object
 *
 * Arguments: 
 *  None
 *
 * Return: 
 *  None
 *
 * Operation:
 *  - if pixel buffer object is already initialize, delete it
 *  - create new pixel buffer object and calculate new grid size.
 *  
 * Revision History:
 *  Mar. 07, 2012      Nnoduka Eruchalu     Initial Revision
 *  Mar. 15, 2014      Nnoduka Eruchalu     Cleaned up comments
 */
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
/*
 * main
 * Description:
 *  main loop
 *
 * Arguments: 
 *  command line arguments
 *
 * Return: 
 *  Program exit status
 *
 * Operation:
 *  - use command-line specified CUDA device, else use device with highest 
 *    Gflops/s
 *  - check CUDA device properties and ensure GPU card is capable of OpenGL
 *    and display. If it isn't inform the user and exit.
 *  - inform user of Mouse's User Interface
 *  - initialize GLUT callback functions
 *  - initialize scene to be rendered
 *  - initialize Pixel Buffer Object
 *  - call GLUT's main loop. Program should be good to go from here!
 *  
 * Revision History:
 *  Mar. 07, 2012      Nnoduka Eruchalu     Initial Revision
 *  Mar. 15, 2014      Nnoduka Eruchalu     Cleaned up comments
 */
int main( int argc, char** argv) 
{
    // use command-line specified CUDA device, otherwise use device with
    // highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );
    
    // check CUDA device properties and ensure GPU card meets minimum reqs.
    int device;
    struct cudaDeviceProp prop;
    cudaGetDevice( &device );
    cudaGetDeviceProperties( &prop, device );
    if( !strncmp( "Tesla", prop.name, 5 ) )
    {
        printf("This program needs a card capable of OpenGL and display.\n");
        printf("Please choose a different device with the "
               "-device=x argument.\n");
        cutilExit(argc, argv);
    }
    
    // inform user of UI.
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
