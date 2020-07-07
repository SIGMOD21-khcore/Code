#include <bits/stdc++.h>
#include <omp.h>
#include <sys/time.h>

using namespace std;

typedef pair <int, int> P;
typedef long long LL;

const int Maxn = 1e4 + 5;
const int Par_num = 16;

int cal_Hindex(vector <int> vec){
    vector <int> bucket;
    for (int i = 0; i <= vec.size(); i++)
        bucket.push_back(0);
    for (int i = 0; i < vec.size(); i++)
        if (vec[i] <= vec.size()) bucket[vec[i]]++;
        else bucket[vec.size()]++;
    int cnt = 0;
    for (int i = vec.size(); i >= 0; i--){
        cnt += bucket[i];
        if (cnt >= i) return i;
    }
}

struct graph{
    vector <int> edge[Maxn], h_nb[Maxn];
    unordered_set <int> edge2[Maxn];
    unordered_set <int> degree_bucket[Maxn];
    int n, m, h_deg[Maxn], deg[Maxn], h, coreness[Maxn], update[Maxn], HI[2][Maxn];
    int LB1[Maxn], LB2[Maxn], LB3[Maxn], setLB[Maxn], UB[Maxn], my_coreness[Maxn],
     del_coreness[Maxn], add_coreness[Maxn], add_LB[Maxn];
    void init(int x){
        for (int i = 1; i <= x; i++){
            edge[i].clear();
            edge2[i].clear();
        }
    }
    void add_edge(int x, int y){
        edge[x].push_back(y);
        edge[y].push_back(x);
    }
    void add_edge2(int x, int y){
        edge2[x].insert(y);
        edge2[y].insert(x);
    }
    void init_edge2(){
        for (int u = 1; u <= n; u++){
            for (int i = 0; i < edge[u].size(); i++){
                int v = edge[u][i];
                if (u < v) add_edge2(u, v);
            }
        }
    }
    void input(){
        scanf("%d %d", &n, &m);
        init(n);
        for (int i = 1; i <= m; i++){
            int x, y;
            scanf("%d %d", &x, &y);
            add_edge(x, y);
            //add_edge2(x, y);
        }
    }
    void output(){
        printf("%d %d\n", n, m);
        for (int i = 1; i <= n; i++){
            for (int j = 0; j < edge[i].size(); j++)
                if (edge[i][j] > i)
                    printf("%d %d\n", i, edge[i][j]);
        }
    }
    void bfs(int x, int h){
        h_nb[x].clear(), h_deg[x] = 0;
        queue <P> q;
        unordered_map <int, bool> vis;
        int cnt = 0;
        q.push(P(x, 1));
        vis[x] = true;
        while (!q.empty()){
            P tmp = q.front();
            q.pop();
            int u = tmp.first, dist = tmp.second;
            for (int i = 0; i < edge[u].size(); i++){
                int v = edge[u][i];
                //cout << u << " " << v << endl;
                if (vis.find(v) != vis.end()) continue;
                vis[v] = true;
                h_deg[x]++, h_nb[x].push_back(v);
                if (dist <= h - 1) q.push(P(v, dist + 1));
            }
        }
    }
    void cal_deg(int H){
        h = H;
        for (int i = 0; i <= n; i++){
            h_nb[i].clear(), h_deg[i] = 0, coreness[i] = -1;
        }
        #pragma omp parallel for num_threads(Par_num)
        for (int i = 1; i <= n; i++){
            bfs(i, h);
            HI[0][i] = h_deg[i];
            HI[1][i] = h_deg[i];
        }
    }

    int Hi_computation(int x, int round){
        unordered_map <int, int> id;
        vector <int> AH, val[2];
        for (int i = 0; i < h_nb[x].size(); i++){
            AH.push_back(-1), val[0].push_back(-1), val[1].push_back(-1);
            id[h_nb[x][i]] = i;
        }
        vector <int> vec[2];
        for (int i = 0; i < edge[x].size(); i++){
            int y = edge[x][i];
            AH[id[y]] = val[0][id[y]] = HI[round][y];
            vec[0].push_back(id[y]);
        }
        for (int i = 2; i <= h; i++){
            vec[(i & 1) ^ 1].clear();
            for (int j = 0; j < vec[i & 1].size(); j++){
                int u = vec[i & 1][j];
                for (int k = 0; k < edge[h_nb[x][u]].size(); k++){
                    int v = edge[h_nb[x][u]][k];
                    if (v == x) continue;
                    if (min(val[i & 1][u], HI[round][v]) > AH[id[v]]){
                        val[(i & 1) ^ 1][id[v]] = AH[id[v]] = min(val[i & 1][u], HI[round][v]);
                        vec[(i & 1) ^ 1].push_back(id[v]);
                    }
                }
            }
        }
        return cal_Hindex(AH);
    }
    int Hi_computation_fast(int x, int round){
        vector <int> AH;
        for (int i = 0; i < h_nb[x].size(); i++){
            int y = h_nb[x][i];
            AH.push_back(HI[round][y]);
        }
        return cal_Hindex(AH);
    }
    void Update(int x, int round, int pre_val, int now_val, bool asyn){
        for (int i = 0; i < h_nb[x].size(); i++){
            int  y = h_nb[x][i];
            if (pre_val >= HI[round][y] && now_val < HI[round][y]){
                if (!asyn)
                    update[y] = (round ^ 1) + 1;
                else
                    update[y] = round + 1;
            }
        }
    }
    int core_decomposition(int h, bool asyn, bool opt2, bool opt3,
                           int& Round1, int& Round2, int& Cnt1, int& Cnt2){
        bool tag = true;
        int round = 0;
        for (int i = 1; i <= n; i++) update[i] = round + 1;
        if (opt3){
            while(tag){
                tag = false;
                Round1++;
                #pragma omp parallel for num_threads(Par_num)
                for (int i = 1; i <= n; i++){
                    if (opt2 && !update[i]) continue;
                    if (update[i] == round + 1) update[i] = false;
                    Cnt1++;
                    if (!asyn){
                        HI[round ^ 1][i] = Hi_computation_fast(i, round);
                        if (HI[round ^ 1][i] != HI[round][i])
                            tag = true, Update(i, round, HI[round][i], HI[round ^ 1][i], asyn);
                    }else{
                        int tmp = Hi_computation_fast(i, round);
                        if (tmp != HI[round][i]){
                            Update(i, round, HI[round][i], tmp, asyn);
                            HI[round][i] = tmp, tag = true;
                        }
                    }
                }
                if (!asyn) round ^= 1;
            }
        }
        tag = true;
        for (int i = 1; i <= n; i++) update[i] = round + 1;
        while(tag){
            tag = false;
            Round2++;
            #pragma omp parallel for num_threads(Par_num)
            for (int i = 1; i <= n; i++){
                if (opt2 && !update[i]) continue;
                if (update[i] == round + 1) update[i] = false;
                Cnt2++;
                if (!asyn){
                    HI[round ^ 1][i] = Hi_computation(i, round);
                    if (HI[round ^ 1][i] != HI[round][i])
                        tag = true, Update(i, round, HI[round][i], HI[round ^ 1][i], asyn);
                }else{
                    int tmp = Hi_computation(i, round);
                    if (tmp != HI[round][i]){
                        Update(i, round, HI[round][i], tmp, asyn);
                        HI[round][i] = tmp, tag = true;
                    }
                }
            }
            if (!asyn) round ^= 1;
        }
        for (int i = 1; i <= n; i++){
            my_coreness[i] = HI[round][i];
        }
        return round;
    }
    void delete_edge(vector <P> del_vec){
        bool tag = true;
        int round = 0, ma = 0;
        for (int i = 1; i <= n; i++) update[i] = false;
        for (int j = 0; j < del_vec.size(); j++){
            int x = del_vec[j].first, y = del_vec[j].second;
            update[x] = update[y] = true;
            ma = max(ma, min(my_coreness[x], my_coreness[y]));
        }
        for (int j = 0; j < del_vec.size(); j++){
            int x = del_vec[j].first, y = del_vec[j].second;
            #pragma omp parallel for num_threads(Par_num)
            for (int i = 0; i < h_nb[x].size(); i++){
                if (!update[h_nb[x][i]]) bfs(h_nb[x][i], h);
                update[h_nb[x][i]] = true;
            }
            #pragma omp parallel for num_threads(Par_num)
            for (int i = 0; i < h_nb[y].size(); i++){
                if (!update[h_nb[y][i]]) bfs(h_nb[y][i], h);
                update[h_nb[y][i]] = true;
            }
        }
        #pragma omp parallel for num_threads(Par_num)
        for (int j = 0; j < del_vec.size(); j++){
            int x = del_vec[j].first, y = del_vec[j].second;
            bfs(x, h), bfs(y, h);
        }
        for (int i = 1; i <= n; i++){
            HI[round][i] = my_coreness[i];
        }
        while(tag){
            tag = false;
            #pragma omp parallel for num_threads(Par_num)
            for (int i = 1; i <= n; i++){
                if (!update[i] || my_coreness[i] > ma){
                    //printf("0 ");
                    continue;
                }
                update[i] = false;
                //printf("1 ");
                int tmp = Hi_computation(i, round);
                if (tmp != HI[round][i]){
                    Update(i, round, HI[round][i], tmp, true);
                    HI[round][i] = tmp, tag = true;
                }
            }
        }
        for (int i = 1; i <= n; i++){
            del_coreness[i] = HI[round][i];
        }
    }
    void insert_edge(vector <P> add_vec){
        bool tag = true;
        int round = 0;
        for (int i = 1; i <= n; i++)
            add_LB[i] = my_coreness[i], update[i] = false;
        for (int j = 0; j < add_vec.size(); j++){
            int x = add_vec[j].first, y = add_vec[j].second;
            bfs(x, h - 1), bfs(y, h - 1);
            update[x] = update[y] = true;
        }
        int mi = n, ma = 0;
        for (int j = 0; j < add_vec.size(); j++){
            int x = add_vec[j].first, y = add_vec[j].second;
            #pragma omp parallel for num_threads(Par_num)
            for (int i = 0; i < h_nb[x].size(); i++){
                if (!update[h_nb[x][i]]) bfs(h_nb[x][i], h), update[h_nb[x][i]] = true;
                mi = min(mi, my_coreness[h_nb[x][i]]);
                ma = max(ma, int(h_nb[h_nb[x][i]].size()));
            }
            #pragma omp parallel for num_threads(Par_num)
            for (int i = 0; i < h_nb[y].size(); i++){
                if (!update[h_nb[y][i]]) bfs(h_nb[y][i], h), update[h_nb[y][i]] = true;
                mi = min(mi, my_coreness[h_nb[y][i]]);
                ma = max(ma, int(h_nb[h_nb[y][i]].size()));
            }
        }
        for (int j = 0; j < add_vec.size(); j++){
            int x = add_vec[j].first, y = add_vec[j].second;
            bfs(x, h), bfs(y, h);
            mi = min(mi, my_coreness[x]), mi = min(mi, my_coreness[y]);
            ma = max(ma, int(h_nb[x].size())), ma = max(ma, int(h_nb[y].size()));
        }
        for (int i = 1; i <= n; i++){
            if (my_coreness[i] < mi || my_coreness[i] >= ma)
                HI[round][i] = my_coreness[i], update[i] = false;
            else
                HI[round][i] = h_nb[i].size(), update[i] = true;
        }
        while(tag){
            tag = false;
            #pragma omp parallel for num_threads(Par_num)
            for (int i = 1; i <= n; i++){
                if (!update[i] || HI[round][i] == add_LB[i]) continue;
                update[i] = false;
                int tmp = Hi_computation_fast(i, round);
                if (tmp != HI[round][i]){
                    Update(i, round, HI[round][i], tmp, true);
                    HI[round][i] = tmp, tag = true;
                }
            }
        }
        tag = true;
        for (int i = 1; i <= n; i++) update[i] = true;
        while(tag){
            tag = false;
            #pragma omp parallel for num_threads(Par_num)
            for (int i = 1; i <= n; i++){
                if (!update[i] || HI[round][i] == add_LB[i]){
                    continue;
                }
                update[i] = false;
                int tmp = Hi_computation(i, round);
                if (tmp != HI[round][i]){
                    Update(i, round, HI[round][i], tmp, true);
                    HI[round][i] = tmp, tag = true;
                }
            }
        }
        for (int i = 1; i <= n; i++){
            add_coreness[i] = HI[round][i];
        }
    }
    bool solve(int h, bool asyn, bool opt2, bool opt3){
        init_edge2();
        for (int i = 1; i <= n; i++) my_coreness[i] = -1;
        clock_t t = clock();
        cal_deg(h);
        double t1 = (clock() - t) * 1.0 / CLOCKS_PER_SEC;
        //cout << "ok" << endl;
        //cal_coreness_hLBUB(S);
        double t2 = (clock() - t) * 1.0 / CLOCKS_PER_SEC;
        //cout << "ok" << endl;
        int Round1 = 0, Round2 = 0, Cnt1 = 0, Cnt2 = 0;
        int round = core_decomposition(h, asyn, opt2, opt3,
                                       Round1, Round2, Cnt1, Cnt2);
        double t3 = (clock() - t) * 1.0 / CLOCKS_PER_SEC;
        printf("%.2f %.2f %.2f\n", t1, t2 - t1, t3 - t2 + t1);
        printf("%d %d %d %d\n", Round1, Cnt1, Round2, Cnt2);
        bool tag = true;
//        for (int i = 1; i <= n; i++){
//            //printf("%d %d %d\n", i, coreness[i], HI[round][i]);
//            if (coreness[i] != my_coreness[i]) tag = false;
//        }
//        if (!tag){
//            for (int i = 1; i <= n; i++){
//                printf("%d %d %d\n", i, coreness[i], HI[round][i]);
//                //if (coreness[i] != my_coreness[i]) tag = false;
//            }
//        }
        return tag;
    }
    bool solve2(int h, bool asyn, bool opt2, bool opt3, int T){
        while(T--){
            cal_deg(h);
            int Round1 = 0, Round2 = 0, Cnt1 = 0, Cnt2 = 0;
            int round = core_decomposition(h, asyn, opt2, opt3,
                                       Round1, Round2, Cnt1, Cnt2);
            vector <P> del_vec;
            while(del_vec.size() < T + 1){
                int x = rand() * 1009 % n + 1;
                while (edge[x].size() == 0) x = rand() * 1009 % n + 1;
                int y = edge[x][rand() % edge[x].size()];
                del_vec.push_back(P(x, y));
                edge[x].erase(remove(edge[x].begin(), edge[x].end(), y), edge[x].end());
                edge[y].erase(remove(edge[y].begin(), edge[y].end(), x), edge[y].end());
            }
            clock_t t = clock();
            delete_edge(del_vec);
            double t1 = (clock() - t) * 1.0 / CLOCKS_PER_SEC;
            cal_deg(h);
            Round1 = 0, Round2 = 0, Cnt1 = 0, Cnt2 = 0;
            round = core_decomposition(h, asyn, opt2, opt3,
                                       Round1, Round2, Cnt1, Cnt2);
            for (int i = 1; i <= n; i++){
                if (my_coreness[i] != del_coreness[i]) return false;
            }
            double t2 = (clock() - t) * 1.0 / CLOCKS_PER_SEC;
            printf("%.2f %.2f\n", t1, t2 - t1);
            for (int j = 0; j < del_vec.size(); j++){
                int x = del_vec[j].first, y = del_vec[j].second;
                edge[x].push_back(y);
                edge[y].push_back(x);
            }
        }
        return true;
    }
    bool solve3(int h, bool asyn, bool opt2, bool opt3, int T){
        while(T--){
            cal_deg(h);
            int Round1 = 0, Round2 = 0, Cnt1 = 0, Cnt2 = 0;
            int round = core_decomposition(h, asyn, opt2, opt3,
                                       Round1, Round2, Cnt1, Cnt2);
            vector <P> add_vec;
            while(add_vec.size() < T + 1){
                int x = rand() * 1009 % n + 1, y = rand() * 1009 % n + 1;
                while (x == y || find(edge[x].begin(), edge[x].end(), y) != edge[x].end())
                    x = rand() * 1009 % n + 1, y = rand() * 1009 % n + 1;
                add_vec.push_back(P(x, y));
                edge[x].push_back(y);
                edge[y].push_back(x);
            }
            clock_t t = clock();
            insert_edge(add_vec);
            double t1 = (clock() - t) * 1.0 / CLOCKS_PER_SEC;
            cal_deg(h);
            Round1 = 0, Round2 = 0, Cnt1 = 0, Cnt2 = 0;
            round = core_decomposition(h, asyn, opt2, opt3,
                                       Round1, Round2, Cnt1, Cnt2);
            for (int i = 1; i <= n; i++){
                //printf("%d %d %d\n", i, my_coreness[i], add_coreness[i]);
                if (my_coreness[i] != add_coreness[i]) return false;
            }
            double t2 = (clock() - t) * 1.0 / CLOCKS_PER_SEC;
            printf("%.2f %.2f\n", t1, t2 - t1);
            for (int j = 0; j < add_vec.size(); j++){
                int x = add_vec[j].first, y = add_vec[j].second;
                edge[x].erase(remove(edge[x].begin(), edge[x].end(), y), edge[x].end());
                edge[y].erase(remove(edge[y].begin(), edge[y].end(), x), edge[y].end());
            }
        }
        return true;
    }
}G;

int main()
{
    //freopen("random_small_v3.txt", "r", stdin);
    //freopen("fb.txt", "r", stdin);
    //freopen("fb.out", "w", stdout);
    int T;
    scanf("%d", &T);
    for (int cas = 1; cas <= T; cas++){
        G.input();
        for (int h = 2; h <= 5; h++){
            //for (S = 1; S <= 256; S*= 4)
            if (G.solve(h, true, true, true))
                cout << "Yes" << endl;
                //continue;
            else{
                printf("h = %d\n", h);
                G.output();
                return 0;
            }
        }
    }
    return 0;
}

/*
2

14 20
1 2
1 6
2 3
3 4
3 5
3 9
4 7
5 8
6 8
7 8
8 10
9 10
9 11
9 12
10 12
10 13
10 14
11 12
12 13
13 14

16 39
1 2
1 16
1 7
1 9
1 11
2 3
2 4
2 16
2 12
2 15
2 7
2 13
2 5
3 5
3 6
3 8
3 7
3 10
3 12
4 12
4 16
4 9
4 13
5 7
5 9
5 10
5 8
5 12
6 13
6 12
7 15
7 9
8 11
9 10
10 12
10 14
11 14
11 16
12 15

8 11
1 2
2 3
2 8
2 5
3 4
4 5
5 6
5 8
5 7
6 7
7 8

8 11
1 2
1 7
2 3
3 4
3 7
4 5
4 7
5 6
5 7
6 7
7 8

9 12
1 2
1 8
2 3
3 4
3 7
3 8
4 5
5 6
6 7
7 8
7 9
8 9

8 11
1 2
1 7
2 3
3 4
3 7
4 5
4 7
5 6
5 7
6 7
7 8

9 11
1 2
1 6
2 3
3 4
4 5
5 6
5 9
6 7
6 9
7 8
8 9

7 14
1 2
2 3
3 4
4 5
5 6
6 7
3 1
2 4
7 2
5 2
5 1
1 6
4 7
7 5

4 5
1 2
2 3
3 4
4 1
2 4
*/
