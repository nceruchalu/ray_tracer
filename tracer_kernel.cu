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

#ifndef _TRACER_KERNEL_H_
#define _TRACER_KERNEL_H_

#include "cutil_math.h"

typedef struct {
    float4 m[3];
} float3x4;

#define EPSILON 0.00001f
#define INTERSECT_EPSILON 0.01f

#define MAX_RAY_TREE_DEPTH 3
#define MAX_STACK_DEPTH 16     // 2^(MAX_RAY_TREE_DEPTH+1) -1 nodes in BinTree
                                // (+1 empty node at index 0)

#define SPHERE   1  /* object types */
#define PLANE    2
#define BOX      3
#define CYLINDER 4
#define CONE     5

/* CUDA doesn't officially support C++, dont want to fight a battle with
 * classes, so just gonna go ahead and define structs (hopefully efficiently)
 *
 * CUDA also doesn't implement recursion. So have to use a binary tree based
 * stack to implement recursion here. There are two recursive calls in the
 * functions, hence a binary tree
 */

/* material properties come up a lot, so best to define a struct for them */
typedef struct{    // material
    float4 amb;    // material ambient color (RGBA)
    float4 diff;   // material diffuse color
    float4 spec;   // material specular color
    float shiny;   // material shininess
    float n;       // refractive index
    float kr;      // reflection coefficients
    float kt;      // transmission coefficient
} Material;


typedef struct {   // struct defining a generic object
    int type;      // object type -- see defines above
    
    Material mat;  // material properties 
    
    /* defined only if type == SPHERE */
    float3 c;      // sphere centre
    float r;       // sphere radius -- also works for cylinder and cone
    
    /* defined only if type == PLANE */
    float4 pl;     // Plane is defined by equation:A*x + By + Cz+D
                   // So in this case:
                   //    plane.x*x + plane.y*y + plane.z*z + plane.w = 0
    
    /* defined only if type == BOX */
    float3 min;   // min and max vertices define a box... think bout it
    float3 max;
    
    /* defined only if type == CYLINDER OR CONE */
    float ymin;  // ymax-ymin gives length of cylinder
    float ymax;
} Object;


typedef struct {   // struct defining an intersection point's properties
    float3 normal; // normal at intersection point
    float3 p;      // position of intersection
    Material mat;  // material properties here     
} Intersect;


typedef struct {   // Light object
    float4 amb;    // ambient color (RGBA)
    float4 diff;   // diffuse color
    float4 spec;   // specular color
    float3 pos;    // light source position
} Light;


struct Ray {         // Ray for RayTracing
    float3 o;	     // origin
    float3 d;	     // direction
    float t;         // minimum distance travelled so far before intersection
    Intersect pt;    // intersection point
    int intersected; // did ray already intersect object? 1 = yes, 0 =no
    int entering;    // ray will be entering or leaving an object?   
};

typedef struct {     // a BinTree "stack" very specific to this ray tracer
    float4 body[MAX_STACK_DEPTH]; // color at a point
    Ray r[MAX_STACK_DEPTH];       // ray that hit this point
    int size;                     // number of slots in array/ nodes in BinTree 
    int max_level;                // the maximum number of levels
    int level;                    // points to last filled level  
    int top;                      // points to next location to push to...so one
                                  //   above top data element
} Stack4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix



///////////////////////////////////////////////////////////////////////////////
// Variables/CONSTANTS that define the scene
///////////////////////////////////////////////////////////////////////////////
#define NUM_OBJECTS 10
#define NUM_LIGHTS  2

/* objects in scene */
__device__ Object d_object[NUM_OBJECTS];
Object object[NUM_OBJECTS];

/* background color */
__device__ float4 d_bgnd_col;
float4 bgnd_col;

/* lights */
__device__ Light d_light[NUM_LIGHTS];
Light light[NUM_LIGHTS];


// push unto the stack. Assuming that top already points to an empty slot
// return 1 if successful, 0 otherwise
__device__
int pushStack(Stack4 &stack, float4 val) 
{
    if(stack.top >= stack.size)
        return 0;
    stack.body[stack.top++] = val;
    return 1;
}

// pop from stack. Have to decrement top to get to actual data
// return 1 if successful, 0 otherwise
__device__
int popStack(Stack4 &stack, float4 &val) 
{
    if(stack.top <= 0)
        return 0;
    val = stack.body[--stack.top];
    return 1;
}

// copy material properties
__device__
void copyMats(Material &m_dest, Material m_source)
{
    m_dest.amb = m_source.amb;  
    m_dest.diff = m_source.diff; 
    m_dest.spec = m_source.spec;
    m_dest.shiny = m_source.shiny;
    m_dest.n = m_source.n;
    m_dest.kr = m_source.kr;
    m_dest.kt = m_source.kt;
    
    return;
}

// copy ray properties
__device__
void copyRay(Ray &dest, Ray source)
{
    dest.o = source.o;
    dest.d = source.d;
    dest.t = source.t;
    dest.intersected = source.intersected;
    dest.entering = source.entering;
    copyMats(dest.pt.mat, source.pt.mat);
    dest.pt.normal = source.pt.normal;
    dest.pt.p = source.pt.p;
}

// intersect ray with sphere
// http://www.cs.unc.edu/~rademach/xroads-RT/RTarticle.html
// http://fuzzyphoton.tripod.com/howtowrt.htm
__device__
int intersectSphere(Ray &r,          /* incident ray */
                    const Object &s  /* sphere under consideration */)
{
    float3 E0 = s.c-r.o;
    float v = dot(E0, r.d);
    float disc = s.r*s.r - (dot(E0,E0) - v*v);
    if (disc < 0) // check if ray actually intersects sphere
        return 0; // no intersection
    
    // take the closest intersection point that is further than the ray origin
    // and infront of the eye-point (i.e. positive ray direction)
    // that is why we compare t0 and t1 to INTERSECT_EPSILON.
    // Why not 0.0f? Simply because there is a high risk that given our limited
    // precision (in float), after a reflection from a point we find that our
    // ray intersects the same object around the same point when it shouldn't
    // with an infinite precision. By taking our starting point at a reasonable
    // distance but close enough to not cause "gaps" we can avoid some artifacts
    float t0 =  v - sqrt(disc);
    float t1 = v + sqrt(disc);
    float t = 0.0;
    int retVal = 0;
    if ((t0>INTERSECT_EPSILON) && (t0 < t1))
    {
        t = t0;
        retVal = 1;
    }
    if ((t1>INTERSECT_EPSILON) && (t1 < t0))
    {
        t = t1;
        retVal = 1;
    }
    
    // if no intersection, end this
    if (retVal == 0)
        return 0;
        
    // test if t isnt the nearest intersection noted
    if (r.intersected && (t > r.t))
       return 0;

    // this is a legitimate intersection... save the properties
    r.intersected = 1;
    r.t = t;                   // record distance of nearest intersection
    
    r.pt.p = r.o + t*r.d;      // update point of intersection
    r.pt.normal = normalize(r.pt.p-s.c); // update normal at point
    copyMats(r.pt.mat, s.mat); // COPY material properties
    
    return 1;                     // there was an intersection!
}


// intersect ray with a plane
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rayplane_intersection.htm
// equation of a plane through 3 points at:
// http://www.math.oregonstate.edu/home/programs/undergrad/CalculusQuestStudyGuides/vcalc/lineplane/lineplane.html
__device__
int intersectPlane(Ray &r,           /* incident ray */
                   const Object &pln /* plane under consideration */)
{
    float3 Pn = make_float3(pln.pl); // unit normal
    float Vd = dot(r.d,Pn);
    
    if (Vd == 0)  // if Vd == 0 then ray is parallel to plane so no intersection
        return 0;
        
    float V0 =  -(dot(r.o,Pn) + pln.pl.w);
    float t = V0/Vd;
    // test if ray intersects behind eye point (i.e. not in positive ray
    // direction), and if it isnt the nearest intersection noted
    if ((t <= INTERSECT_EPSILON) || (r.intersected && (t > r.t)))
        return 0;
    
    // this is a legitimate intersection
    r.intersected = 1;
    r.t = t;                   // record distance of nearest intersection
    r.pt.p = r.o + t*r.d;      // update point of intersection
    r.pt.normal = normalize(Pn);    // update normal at point of intersection
    copyMats(r.pt.mat, pln.mat); // COPY material properties
    
    return 1;                     // there was an intersection!  
}


// helper for intersectBox
/* return 1 if a point is in a plane, 0 otherwise
 *    to check if a point (px,py,pz) is on a plane
 *    nx*(ppx-px) + ny*(ppy-py) + nz*(ppz-pz) == 0.0
 *      -- note that equality to 0.0 isnt happening so use an EPSILON check
 *      -- that is check if the absolute value is less than EPSILON
 */
__device__
int pointInPlane(float3 n,  /* normal*/
                 float3 pp, /* point_in_plane */
                 float3 p  /* point under consideration */)
{
    int inPlane = 0;
    float3 vecInPlane = p-pp;
    float res = dot(n, vecInPlane);
    if (res < 0.0f)
        res *= -1.0f;
    if (res < EPSILON)
        inPlane = 1;
    
    return inPlane;
}


// intersect a ray with a Box
// http://www.visualization.hpc.mil/wiki/Adding_in_a_Box_Object
__device__
int intersectBox(Ray &r,           /* incident ray */
                   const Object &box /* box under consideration */)
{

    float tmp, tnear = -1.0e6, tfar=1.0e6;
    float3 tmin = (box.min - r.o)/r.d;
    float3 tmax = (box.max - r.o)/r.d;
    if(tmin.x>tmax.x) { tmp=tmin.x; tmin.x=tmax.x; tmax.x=tmp;}
    if(tmin.y>tmax.y) { tmp=tmin.y; tmin.y=tmax.y; tmax.y=tmp;}
    if(tmin.z>tmax.z) { tmp=tmin.z; tmin.z=tmax.z; tmax.z=tmp;}
    
    tnear=max(tmin.z,max(tmin.y,max(tmin.x,tnear)));
    tfar =min(tmax.z,min(tmax.y,min(tmax.x,tfar )));
    if(tnear>tfar) return 0; // The box is missed.
    if(tfar<INTERSECT_EPSILON) return 0; // The box is behind us.
    
    if(tnear<INTERSECT_EPSILON) {return 0; } // We are inside the box.
    // have an intersection
    float t = tnear;
        
    // check this is nearest intersection noted
    if (r.intersected && (t>r.t))
        return 0;
    
    // this is a legitimate intersection
    r.intersected = 1;
    r.t = t;                   // record distance of nearest intersection
    r.pt.p = r.o + t*r.d;      // update point of intersection
    
    // update normal at point of intersection
    float3 distance = box.min - r.pt.p;
    float min_dist=abs(distance.x);
    int min=0;
    if(abs(distance.y) < min_dist) { min_dist=abs(distance.y); min=2; }
    if(abs(distance.z) < min_dist) { min_dist=abs(distance.z); min=4; }
    distance = box.max - r.pt.p;
    if(abs(distance.x) < min_dist) { min_dist=abs(distance.x); min=1; }
    if(abs(distance.y) < min_dist) { min_dist=abs(distance.y); min=3; }
    if(abs(distance.z) < min_dist) { min_dist=abs(distance.z); min=5; }
 
    r.pt.normal = make_float3(0, 0, 1);
    if (min==0) {r.pt.normal = make_float3(-1, 0, 0);}
    if (min==1) {r.pt.normal = make_float3( 1, 0, 0);}
    if (min==2) {r.pt.normal = make_float3( 0,-1, 0);}
    if (min==3) {r.pt.normal = make_float3( 0, 1, 0);}
    if (min==4) {r.pt.normal = make_float3( 0, 0,-1);}
        
    if (dot(r.pt.normal, r.d) > 0) // normal and ray must be in opposite dirs
        r.pt.normal *= -1;
    
    copyMats(r.pt.mat, box.mat); // COPY material properties
    
    return 1;                     // there was an intersection!  
}



// intersect ray with a cylinder (capless!)
// http://www.visualization.hpc.mil/wiki/Adding_in_a_Cylinder_Object
__device__
int intersectCylinder(Ray &r,           /* incident ray */
                      const Object &cyl /* cylinder under consideration */)
{    
    float3 pvt_center = make_float3(0.0, (cyl.ymin+cyl.ymax)/2, 0.0);
    float3 Rd=r.d;
    float3 Ro=pvt_center - r.o;
    float3 pnt_intrsct;
    float a = Rd.x*Rd.x + Rd.z*Rd.z;
    float b = Ro.x*Rd.x + Ro.z*Rd.z;
    float c = Ro.x*Ro.x + Ro.z*Ro.z - (cyl.r*cyl.r);
    float disc = b*b - a*c;
    float t, d, root1, root2;
    int return_value = 0;
    
    // If the discriminant is less than 0, then we totally miss the cylinder.
    if (disc > 0.0)
    {
        d = sqrt(disc);
        root2 = (b + d)/a;
        root1 = (b - d)/a;
        // If root2 < 0, then root1 is also < 0, so they are both misses.
        if (root2 > INTERSECT_EPSILON)
        {
            // If root2 > 0, and root1 < 0, we are inside the cylinder.
            if(root1 < INTERSECT_EPSILON)
            {
                return_value=0;
                // If root2 > 0, and root1 > 0, we are hit the cylinder.
            } else {
             t=root1; return_value= 1;
           }
         }
        pnt_intrsct = r.o + Rd*t ;
        // Limit the y values
        if((pnt_intrsct.y>cyl.ymax)||(pnt_intrsct.y<cyl.ymin)) {
            pnt_intrsct = r.o + Rd*root2 ;
            t = root2;
            // Are we too high in our first hit, but hit the back wall later
            if((pnt_intrsct.y>cyl.ymax)||(pnt_intrsct.y<cyl.ymin)) {
                return_value = 0;
            }
        }
    }
        
    if (!return_value)
        return 0;
    // have intersection
    // check this is nearest intersection noted
    if (r.intersected && (t>r.t))
        return 0;
    
    // this is a legitimate intersection
    r.intersected = 1;
    r.t = t;                   // record distance of nearest intersection
    r.pt.p = pnt_intrsct;      // update point of intersection

    //update normal
    r.pt.normal = normalize(r.pt.p - make_float3(0.0,r.pt.p.y,0.0)); 
           
    copyMats(r.pt.mat, cyl.mat); // COPY material properties
    
    return 1;                     // there was an intersection!  
}
    

// intersect ray with a cone (capless!)
// http://www.visualization.hpc.mil/wiki/Adding_in_a_Cone_Object
__device__
int intersectCone(Ray &r,           /* incident ray */
                      const Object &con /* cone under consideration */)
{
    float3 pvt_center = make_float3(0.0, con.ymin, 0.0);
    float pvt_height = con.ymax-con.ymin;
    float pvt_e = -(con.r*con.r)/(pvt_height*pvt_height);
    float3 Rd=r.d;
    float3 Ro=r.o;
    float3 omc=pvt_center - Ro;
    omc.y = pvt_center.y - Ro.y + pvt_height;
    float3 pnt_intrsct;
    float a = Rd.x*Rd.x + pvt_e*Rd.y*Rd.y + Rd.z*Rd.z;
    float b = omc.x*Rd.x + pvt_e*omc.y*Rd.y + omc.z*Rd.z;
    float c = omc.x*omc.x + pvt_e*omc.y*omc.y + omc.z*omc.z;
    float disc = b*b - a*c;
    float t, d, root1, root2;
    int return_value = 0;

    // If the discriminant is less than 0, then we totally miss the cone.
    if (disc > 0.0) {
        d = sqrt(disc);
        root2 = (b + d)/a;
        root1 = (b - d)/a;
        // If root2 < 0, then root1 is also < 0, so they are both misses.
        if (root2 > INTERSECT_EPSILON) {
            // If root2 > 0, and root1 < 0, we are inside the cone.
            if(root1 < INTERSECT_EPSILON) {
                return_value=0;
                // If root2 > 0, and root1 > 0, we are hit the cone.
            } else {
               t=root1; return_value= 1;
            }
        }
        pnt_intrsct = Ro + Rd*t ;
        
        // Limit the y values: ymin <= y <= ymax
        // If the point of intersection is too low or too high, record it as a
        // miss.
        if((pnt_intrsct.y>(pvt_center.y+pvt_height))||
           (pnt_intrsct.y<pvt_center.y)) {
            pnt_intrsct = Ro + Rd*root2 ;
            t = root2;
            // Are we too high in our first hit, but hit the back wall later
            if((pnt_intrsct.y>(pvt_center.y+pvt_height))||
               (pnt_intrsct.y<pvt_center.y)) {
                return_value = 0;
            }
        }
    }
    
    if (!return_value)
        return 0;
    // have intersection
    // check this is nearest intersection noted
    if (r.intersected && (t>r.t))
        return 0;
    
    // this is a legitimate intersection
    r.intersected = 1;
    r.t = t;                   // record distance of nearest intersection
    r.pt.p = pnt_intrsct;      // update point of intersection

    //update normal
    a = r.pt.p.x - pvt_center.x;
    b = r.pt.p.y - pvt_center.y - pvt_height;
    c = r.pt.p.z - pvt_center.z;
    r.pt.normal=normalize(make_float3(a,pvt_e*b,c));
    
    copyMats(r.pt.mat, con.mat); // COPY material properties
    
    return 1;                     // there was an intersection! 

}

    
// get first intersection of a ray in the scene.
__device__
void get_first_intersection(Ray &r)
{
    // this loop goes through all objects in our space and at the end
    // the Ray is updated with the information of the nearest/first
    // intersection
    for(int i=0; i < NUM_OBJECTS; i++)
    {
        if(d_object[i].type == SPHERE)          // call the right intersect
            intersectSphere(r, d_object[i]);    // function for each object type
        else if(d_object[i].type == PLANE)
            intersectPlane(r, d_object[i]);
        else if(d_object[i].type == BOX)
            intersectBox(r, d_object[i]);
        else if(d_object[i].type == CYLINDER)
            intersectCylinder(r, d_object[i]);
        else if(d_object[i].type == CONE)
            intersectCone(r, d_object[i]);
    }
    return; /* intersected object info is contained in ray */
}


// get color at a point
// if the point is in shadow, 
__device__
float4 get_point_color(Ray &r)
{
    float4 col4 = make_float4(0.0f);  // will accumulate the colors
    
    // the amount of diffuse+specular color to be accumulated is scaled by
    // the shadow variable. Start by assuming this point is not in a shadow
    float shadow = 1.;    
    
    if (!r.intersected)   // ray didnt intersect an object, so use background
        return d_bgnd_col;

    // need to add in the contributions from all the light sources
    for(int i = 0; i< NUM_LIGHTS; i++)
    {
        float3 lightDir = normalize(d_light[i].pos -r.pt.p); // from p to light
        col4 += r.pt.mat.amb * d_light[i].amb; // add in ambient color
        
        /* check if point is in shadow for this light source */
        Ray shadowCheck;                // create a shadow ray from intersection
        shadowCheck.o = r.pt.p;         // pt to light, then check if it
        shadowCheck.d = lightDir;       // intersects any objects in the scene.
        shadowCheck.intersected = 0;    // if it does, then shadow factor is set
        get_first_intersection(shadowCheck); // to transmission coefficient  of
        if(shadowCheck.intersected)    // that object. If object is opaque,
            shadow = shadowCheck.pt.mat.kt; // shadow factor becomes 0
        
        
        /* compute the dot product between normal and normalized lightdir */
        float NdotL = max(dot(r.pt.normal,lightDir), 0.0);
        float3 R = 2*r.pt.normal*NdotL - lightDir; // R = light ray's reflection
        R = normalize(R);
        float3 Eye = normalize(-r.d);
        /* compute the dot product between reflection and eye-view */
        float RdotE = max(dot(R,Eye),0.0);
	
        if (NdotL > 0.0)
        {   
            /* add diffuse component */
            col4 += shadow*(r.pt.mat.diff * d_light[i].diff * NdotL);
        }
        /* add specular component */
        col4 += shadow*(r.pt.mat.spec * pow(RdotE, r.pt.mat.shiny));
    }
    return col4;        
}

// initialize_stack by setting up the root node and other properties of the
// stack. Also setup all rays to a non-intersected state and set all colors
// to blank. This way we can just loop through and add up colors easily.
// Note that the root is initialized at index 1. This is just how
// array based binary trees work. index 0 is purposely left blank.
__device__
void initStack(Stack4 &stack, float4 val, Ray &r)
{
    stack.size = MAX_STACK_DEPTH;
    stack.max_level = MAX_RAY_TREE_DEPTH;   
    stack.top = 2;                          
    stack.level = 0;
    
    for(int i=0; i < stack.size ;i++)
    {
        stack.body[i] = make_float4(0.);
        stack.r[i].intersected = 0;
    }
    stack.body[1] = val;
    copyRay(stack.r[1], r);
}

// This function to update the next level is very specific to the ray tracing
// algorithm.
// the root of the stack is already initialized.
// Remember that level is setup to point to the currently filled one. So
// when this function is called you plan on filling level+1
// At each level the array indices will run from
//      2^stack.level to [2^(stack.level+ 1) -1]
//
// When done updating the level pointer (INC it)
// 
// Algorithm: 
// you get the stack with the level of the last filled level, using this, you
// can update the next level, with this basic idea.
// have a for loop of index i going from 2^level to 2^(level+1) - 1
// at each entry here, if it's value is non-zero then we update its
// reflection(left) and refraction(right) child nodes.
__device__ 
int updateNextLevel(Stack4 &stack)
{
    int start = powf(2,stack.level);
    int stop = 2*start -1;
    stack.top = stop+1;  // update top of stack appropriately -- next empty slot
    int child_nodes = 0; // boolean value indicating existence of child nodes
    float kr, kt, n, c1, c2, sinT2;
    
    
    if(stack.level >= stack.max_level) // check if stack is full
        return 0;
    
    for(int i = start; i <= stop; i++) //loop through all nodes on current level
    {
        Ray r;                        // dont want to keep on accessing
        copyRay(r, stack.r[i]);       // an array, so copy ray info
        
        if(!r.intersected)    // if ray didnt intersect point, move on
            continue;
        
        kr = r.pt.mat.kr;    // copy these values for easier typing
        kt = r.pt.mat.kt;
        
        // save reflected ray and color from the ray if object is reflective
        if(kr > 0.)
        {
            child_nodes = 1;   // next level actually got updated
            stack.r[2*i].o = r.pt.p;
            stack.r[2*i].d = normalize(r.d -2*r.pt.normal*dot(r.pt.normal,r.d));
            stack.r[2*i].intersected = 0;
            
            get_first_intersection(stack.r[2*i]);
            
            stack.r[2*i].pt.mat.kr *= kr; // need to recursively multiply kr
            stack.body[2*i] = kr*get_point_color(stack.r[2*i]);
        }
        else 
        {
            stack.r[2*i].pt.mat.kr = 0.0;
        }
        
        // save refracted ray and color from the ray if object is non-opaque
        if(kt > 0.)
        {
            child_nodes = 1; // next level actually got updated
            
            // refractive index value n, depends on direction( in or out) of ray
            // which is flipped each time a refracted ray is created
            n = (r.entering ? 1./r.pt.mat.n : r.pt.mat.n);
            stack.r[2*i+1].entering = 1- r.entering; // flip boolean value
            c1 = -dot(r.pt.normal, r.d);
            c2 = sqrt(1- n*n*(1 - c1*c1));
            sinT2 = n*n*(1 - c1*c1);
            
            if (sinT2 > 1.0) { // total internal reflection -- so use reflection
                               // code for ray direction
                stack.r[2*i+1].d = normalize(r.d -2*r.pt.normal*dot(r.pt.normal,r.d));
            } else {
                stack.r[2*i+1].d = normalize((n*r.d) + (n*c1 -c2)*r.pt.normal);
            }
            
            stack.r[2*i+1].o = r.pt.p;
            
            
            stack.r[2*i+1].intersected = 0;
            
            get_first_intersection(stack.r[2*i+1]);
            
            stack.r[2*i+1].pt.mat.kt *= kt; // recursively multiply kt
            stack.body[2*i+1] = kt*get_point_color(stack.r[2*i+1]);
            
            
        }
        else 
        {
            stack.r[2*i].pt.mat.kt = 0.0;
        }
    }
    
    if(child_nodes)       
        stack.level++;      // if there was a child node
    
    return child_nodes;
}


// trace a ray
// http://www.cs.unc.edu/~rademach/xroads-RT/RTarticle.html
__device__
float4 trace_ray(Ray r)
{
    float4 point_color;
    float4 final_color = make_float4(0.); // final point color
    Stack4 stack;
        
    // keep looping while ray is still intersecting and there is still a color
    // to push
    get_first_intersection(r);
    point_color = get_point_color(r);
    initStack(stack, point_color, r);
   
    while(updateNextLevel(stack))  // recursion not available on CUDA
        continue;                  // so keep on updating next level in BinTree
                                   // stack until done or hit max depth
    
    while(popStack(stack, point_color)) // now just get pop off all saved
        final_color+= point_color;      // colors and accumulate them
    
    return final_color;                 // then return final pixel color
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

// color conversion functions
__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__device__ uint rFloatToInt(float r)
{
    r = __saturatef(r);   // clamp to [0.0, 1.0]
    return (uint(r*255)<<24) | (uint(r*255)<<16) | (uint(r*255)<<8) | uint(r*255);
}

// perform volume rendering
__global__ void
d_render(uint *d_output, uint imageW, uint imageH)
{
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    
    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix,
                               make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);
    
    eyeRay.intersected = 0;   // obviously hasnt intersected any object yet.
    eyeRay.entering = 1;      // starts off entering objects
    
    if ((x < imageW) && (y < imageH)) {
        // write output color
        uint i = __umul24(y, imageW) + x;

        // trace ray and determine pixel color
        float4 col4 = trace_ray(eyeRay);

        d_output[i] = rgbaFloatToInt(col4);
    }
}

#endif                                              // #ifndef _TRACER_KERNEL_H_
