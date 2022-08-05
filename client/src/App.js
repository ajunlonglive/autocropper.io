import './App.scss';
import 'boxicons/css/boxicons.min.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import AppLayout from './components/layout/AppLayout';
import Blank from './pages/Blank';
import Upload from './pages/Upload';
import HomePage from './pages/HomePage';
import About from './pages/About';

function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path='/' element={<AppLayout />}>
                    <Route index element={<HomePage />} />
                    <Route path='/upload' element={<Upload />} />
                    <Route path='/about' element={<About />} />
                    <Route path='/contact' element={<Blank />} />
                </Route>
            </Routes>
        </BrowserRouter>
    );
}

export default App;