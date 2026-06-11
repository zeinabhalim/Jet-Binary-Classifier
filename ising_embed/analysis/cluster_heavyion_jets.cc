#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"

using namespace std;
using namespace fastjet;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <input.oscar> <output.csv> <label>" << endl;
        return 1;
    }

    string input_path = argv[1];
    string output_path = argv[2];
    int label = atoi(argv[3]);

    double R = 0.4;
    double pt_min = 5.0;
    double pt_min_const = 0.15;
    JetDefinition jet_def(antikt_algorithm, R);

    ifstream file(input_path);
    ofstream out(output_path);
    out << "label,event,jet_pt,jet_eta,jet_phi,jet_mass,"
        << "nconstituents,pt_d,sqrt_d12" << endl;

    int event_num = 0, jet_count = 0;
    string line;

    while (file.peek() != EOF) {
        // Find next event
        bool found = false;
        while (getline(file, line))
            if (line.find("event") != string::npos && line.find("end") == string::npos)
                { found = true; break; }
        if (!found) break;

        vector<PseudoJet> parts;
        while (getline(file, line)) {
            if (line.empty()) continue;
            if (line.find("end") != string::npos) break;
            if (line[0] == '#') continue;
            istringstream iss(line);
            double t,x,y,z,m,p0,px,py,pz; int pdg,id,ch;
            if (!(iss>>t>>x>>y>>z>>m>>p0>>px>>py>>pz>>pdg>>id>>ch)) continue;
            if (sqrt(px*px+py*py) > pt_min_const)
                parts.push_back(PseudoJet(px,py,pz,p0));
        }
        if (parts.size() < 2) { event_num++; continue; }

        ClusterSequence cs(parts, jet_def);
        auto jets = sorted_by_pt(cs.inclusive_jets(pt_min));

        for (auto& jet : jets) {
            if (jet.pt() < pt_min) continue;
            auto cs2 = jet.constituents();
            double sum_pt=0, sum_pt2=0;
            for (auto& c : cs2) { double pt=c.pt(); sum_pt+=pt; sum_pt2+=pt*pt; }

            double sqrt_d12 = 0;
            if (cs2.size() >= 2) {
                JetDefinition ca(cambridge_algorithm, R);
                ClusterSequence csca(cs2, ca);
                auto ca_jets = csca.exclusive_jets_up_to(2);
                if (ca_jets.size() == 2)
                    sqrt_d12 = sqrt(ca_jets[0].kt_distance(ca_jets[1]));
            }

            out << label << "," << event_num << ","
                << jet.pt() << "," << jet.eta() << "," << jet.phi() << ","
                << max(jet.m(),0.0) << "," << (int)cs2.size() << ","
                << sqrt(sum_pt2)/sum_pt << "," << sqrt_d12 << endl;
            jet_count++;
        }
        event_num++;
        if (event_num % 100 == 0)
            cout << "  " << event_num << " events -> " << jet_count << " jets" << endl;
    }
    file.close(); out.close();
    cout << "Done: " << event_num << " events, " << jet_count << " jets -> " << output_path << endl;
    return 0;
}

