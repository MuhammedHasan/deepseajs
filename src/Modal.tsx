// @ts-ignore
import { DNALogo } from 'logojs-react';


const Modal = ({ ppm, isOpen, setIsOpen }: { ppm: number[][], isOpen: boolean, setIsOpen: (a: boolean) => void }) => {

    return (
        <>
            {isOpen && (
                <div className="cursor-default backdrop-blur-sm fixed inset-0 z-10 w-screen overflow-y-auto">
                    <div className="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
                        <div className="relative transform overflow-hidden rounded-lg bg-white text-left shadow-xl transition-all sm:my-8 w-4/5">
                            <div className="bg-white px-4 pb-4 pt-5 sm:p-6 sm:pb-4">
                                <div className="sm:flex sm:items-start grow">
                                    <div className="mt-3 text-center sm:ml-4 sm:mt-0 sm:text-left grow">
                                        <div className='flex justify-between'>
                                            <h3
                                                className="text-base font-semibold leading-6 text-gray-900"
                                                id="modal-title"
                                            >
                                                Gradients to input sequence
                                            </h3>
                                        </div>
                                        <div className="mt-2 h-92">
                                            {
                                                ([...Array(20).keys()]).map((i) =>
                                                    <div key={i}>
                                                        <DNALogo ppm={ppm} />
                                                    </div>
                                                )
                                            }
                                        </div>
                                    </div>
                                    <button
                                        className="inline-block p-3 text-gray-700 hover:bg-gray-50 focus:relative"
                                        title="Close"
                                        onClick={() => setIsOpen(false)}
                                    >
                                        <svg
                                            xmlns="http://www.w3.org/2000/svg"
                                            viewBox="0 0 50 50"
                                            width="28px"
                                            height="28px"
                                        >
                                            <path d="M 7.71875 6.28125 L 6.28125 7.71875 L 23.5625 25 L 6.28125 42.28125 L 7.71875 43.71875 L 25 26.4375 L 42.28125 43.71875 L 43.71875 42.28125 L 26.4375 25 L 43.71875 7.71875 L 42.28125 6.28125 L 25 23.5625 Z" />
                                        </svg>
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}

export default Modal