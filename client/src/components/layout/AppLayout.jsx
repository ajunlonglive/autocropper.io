import { Outlet } from "react-router-dom";
import Sidebar from "../sidebar/Sidebar";
import Footer from "./Footer"

const AppLayout = () => {
    return <div style={{
        padding: '50px 50px 10px 370px'
    }}>
        <Sidebar />
        <Outlet />
        {/* <Footer /> */}

    </div>;
};

export default AppLayout;
