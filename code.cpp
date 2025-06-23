#include<bits/stdc++.h>

#define ll long long
#define mk make_pair
#define fi first
#define se second

using namespace std;

#define gc getchar
inline int read(){
    int x=0,f=1;char c=getchar();
    for(;(c<'0'||c>'9');c=getchar()){if(c=='-')f=-1;}
    for(;(c>='0'&&c<='9');c=getchar())x=x*10+(c&15);
    return x*f;
}

const int mod=998244353;
int ksm(int x,ll y,int p=mod){
    int ans=1;y%=(p-1);
    for(int i=y;i;i>>=1,x=1ll*x*x%p)if(i&1)ans=1ll*ans*x%p;
    return ans%p;
}
int inv(int x,int p=mod){return ksm(x,p-2,p)%p;}
mt19937 rnd(time(0));
int randint(int l,int r){
    uniform_int_distribution<>dist(l,r);
    return dist(rnd);
}
void add(int &x,int v){x+=v;if(x>=mod)x-=mod;}
void Mod(int &x){if(x>=mod)x-=mod;}
int cmod(int x){if(x>=mod)x-=mod;return x;}

template<typename T>void cmax(T &x,T v){x=max(x,v);}
template<typename T>void cmin(T &x,T v){x=min(x,v);}

// change 1
// change 2

signed main(void){

#ifndef ONLINE_JUDGE
    freopen("in.txt","r",stdin);
#endif

    return 0;
}