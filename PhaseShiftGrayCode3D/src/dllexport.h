#pragma once

#define JSLAB_Algorithm_Dual

#ifdef JSLAB_Algorithm_Dual
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __declspec(dllimport)
#endif

